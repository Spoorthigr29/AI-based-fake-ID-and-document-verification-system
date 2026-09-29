from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from django.core.exceptions import ValidationError

from .models import Document, DocumentQualityAnalysis
from .validators import validate_uploaded_document, validate_uploaded_selfie
from .services.quality_analyzer import DocumentQualityAnalyzer
from .services.document_classifier import DocumentClassifier
from verification.services.verification_pipeline import VerificationPipelineOrchestrator
from audit.models import AuditLog


class StartVerificationPipelineAPIView(APIView):
    """
    POST /api/verification/start/<verification_id>/
    Orchestrates the entire 12-stage multi-modal forensic screening pipeline:
    1. Validate document
    2. Analyze quality
    3. Classify document
    4. Run OCR
    5. Extract fields
    6. Detect face
    7. Compare selfie and document face
    8. Analyze tampering (ELA)
    9. Check identity consistency
    10. Calculate risk score
    11. Generate explanation
    12. Save verification result

    Returns:
    {
      "verification_id": "VX-2026-000001",
      "status": "COMPLETED" | "MANUAL_REVIEW",
      "risk_score": 18,
      "category": "LOW_RISK" | "MANUAL_REVIEW" | "HIGH_RISK"
    }
    """

    def post(self, request, verification_id, *args, **kwargs):
        try:
            document = Document.objects.filter(verification_id=verification_id).first()
            if not document and verification_id.isdigit():
                document = Document.objects.filter(pk=int(verification_id)).first()

            if not document:
                return Response({
                    "error": f"Verification case '{verification_id}' not found."
                }, status=status.HTTP_404_NOT_FOUND)

            intake_data = request.data if isinstance(request.data, dict) else {}

            # Audit log start of verification
            AuditLog.log_event(
                action=AuditLog.ACTION_VERIFICATION_START,
                user=request.user,
                document=document,
                metadata={"verification_id": document.verification_id}
            )

            orchestrator = VerificationPipelineOrchestrator()
            result = orchestrator.run_pipeline(document, intake_data=intake_data)

            # Map category to standard string format
            category = result.get("category", "LOW_RISK")

            return Response({
                "verification_id": result["verification_id"],
                "status": result["status"],
                "risk_score": result["risk_score"],
                "category": category,
                "stages": result.get("stages", []),
                "explanation": result.get("explanation", ""),
                "duration_ms": result.get("duration_ms", 0)
            }, status=status.HTTP_200_OK)

        except FileNotFoundError as fnf:
            return Response({
                "error": str(fnf)
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({
                "error": f"Pipeline execution failed: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentUploadAPIView(APIView):
    """
    POST /api/documents/upload/
    Uploads an identity document (and optional applicant selfie),
    validates binary content, calculates SHA-256 digests, and generates a verification case.
    """
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        doc_file = request.FILES.get('original_file') or request.FILES.get('document') or request.FILES.get('identity_document')
        selfie_file = request.FILES.get('applicant_selfie') or request.FILES.get('selfie_file') or request.FILES.get('selfie')
        doc_type = request.data.get('document_type', Document.DOC_TYPE_OTHER)

        if not doc_file:
            return Response({
                "success": False,
                "error": "Missing identity document file. Please attach 'original_file' or 'document'."
            }, status=status.HTTP_400_BAD_REQUEST)

        # Multi-layer validation
        try:
            validate_uploaded_document(doc_file)
            if selfie_file:
                validate_uploaded_selfie(selfie_file)
        except ValidationError as e:
            return Response({
                "success": False,
                "error": e.message if hasattr(e, 'message') else str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({
                "success": False,
                "error": f"Invalid file content: {str(e)}"
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            document = Document.objects.create(
                document_type=doc_type,
                original_file=doc_file,
                selfie_file=selfie_file,
                processing_status=Document.STATUS_UPLOADED,
                uploaded_by=request.user if request.user.is_authenticated else None
            )

            # Automatically execute Initial Quality & Classification screening
            file_path = document.original_file.path if hasattr(document.original_file, 'path') else document.original_file
            quality_data = DocumentQualityAnalyzer.analyze_quality(file_path)
            class_data = DocumentClassifier.classify(file_path)

            doc_status = Document.STATUS_MANUAL_REVIEW if quality_data["is_poor_quality"] else Document.STATUS_UPLOADED
            document.processing_status = doc_status
            document.save(update_fields=['processing_status'])

            DocumentQualityAnalysis.objects.create(
                document=document,
                classified_type=class_data["document_type"],
                classification_confidence=class_data["confidence"],
                quality_score=quality_data["quality_score"],
                issues=quality_data["issues"],
                checklist=quality_data["checklist"],
                metrics=quality_data["metrics"],
                is_poor_quality=quality_data["is_poor_quality"]
            )

            # Audit log entry
            AuditLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                action=AuditLog.ACTION_UPLOAD,
                document=document,
                metadata={
                    "verification_id": document.verification_id,
                    "document_type": document.document_type,
                    "classified_type": class_data["document_type"],
                    "quality_score": quality_data["quality_score"],
                    "is_poor_quality": quality_data["is_poor_quality"],
                    "has_selfie": bool(selfie_file),
                }
            )

            return Response({
                "success": True,
                "verification_id": document.verification_id,
                "document_id": document.id,
                "message": "Document uploaded successfully"
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({
                "success": False,
                "error": f"Failed to store document: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentAnalyzeAPIView(APIView):
    """
    POST /api/documents/analyze/<verification_id>/
    Performs document classification and document quality analysis.
    """
    def post(self, request, verification_id, *args, **kwargs):
        try:
            document = Document.objects.get(verification_id=verification_id)
        except Document.DoesNotExist:
            return Response({
                "error": f"Verification case '{verification_id}' not found."
            }, status=status.HTTP_404_NOT_FOUND)

        if not document.original_file:
            return Response({
                "error": "No identity document file attached."
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            file_input = document.original_file.path if hasattr(document.original_file, 'path') else document.original_file
            ocr_text = ""
            if hasattr(document, 'ocr_analysis'):
                ocr_text = document.ocr_analysis.raw_text

            quality_res = DocumentQualityAnalyzer.analyze_quality(file_input)
            class_res = DocumentClassifier.classify(file_input, ocr_text=ocr_text)

            # Update status to MANUAL_REVIEW if quality is poor
            if quality_res["is_poor_quality"]:
                document.processing_status = Document.STATUS_MANUAL_REVIEW
                document.save(update_fields=['processing_status'])

            # Store or update analysis in database
            DocumentQualityAnalysis.objects.update_or_create(
                document=document,
                defaults={
                    "classified_type": class_res["document_type"],
                    "classification_confidence": class_res["confidence"],
                    "quality_score": quality_res["quality_score"],
                    "issues": quality_res["issues"],
                    "checklist": quality_res["checklist"],
                    "metrics": quality_res["metrics"],
                    "is_poor_quality": quality_res["is_poor_quality"]
                }
            )

            # Audit Log
            AuditLog.objects.create(
                user=request.user if request.user.is_authenticated else None,
                action=AuditLog.ACTION_AI_SCAN,
                document=document,
                metadata={
                    "pipeline": "QUALITY_AND_CLASSIFICATION",
                    "classified_type": class_res["document_type"],
                    "quality_score": quality_res["quality_score"],
                    "issues_count": len(quality_res["issues"]),
                    "status_assigned": document.processing_status
                }
            )

            return Response({
                "document_type": class_res["document_type"],
                "quality_score": quality_res["quality_score"],
                "issues": quality_res["issues"]
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                "error": f"Quality analysis failed: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class VerificationDebugAPIView(APIView):
    """
    GET /api/verification/<verification_id>/debug/
    Provides complete forensic telemetry across all 24 verification pipeline stages for audit and diagnostics:
    - document_classifier_result
    - image_quality_score & metrics
    - ocr_confidence & extracted_fields
    - qr_detected, qr_decoded, qr_signature_status, qr_consistency
    - tamper_score & granular region_scores (photo, name, dob, number, background)
    - document_face_detection & selfie_face_detection
    - liveness_status
    - face_similarity & calibrated_threshold (0.70)
    - identity_consistency
    - risk_calculation & contributing factors
    """
    def get(self, request, verification_id, *args, **kwargs):
        document = Document.objects.filter(verification_id=verification_id).first()
        if not document and verification_id.isdigit():
            document = Document.objects.filter(pk=int(verification_id)).first()

        if not document:
            return Response({"error": f"Verification case '{verification_id}' not found."}, status=status.HTTP_404_NOT_FOUND)

        quality_analysis = getattr(document, 'quality_analysis', None)
        face_verification = getattr(document, 'face_verification', None)
        tamper_analysis = getattr(document, 'tamper_analysis', None)
        consistency_check = getattr(document, 'consistency_check', None)
        ocr_analysis = getattr(document, 'ocr_analysis', None)

        # Risk Engine Evaluation
        from risk_engine.services.risk_calculator import DeterministicRiskCalculator
        calculator = DeterministicRiskCalculator()
        risk_result = calculator.evaluate_for_document(document)

        metrics = quality_analysis.metrics if quality_analysis else {}
        qr_data = metrics.get('qr_result', {})
        qr_consistency = metrics.get('qr_consistency', {})
        template_res = metrics.get('template_analysis', {})
        
        tamper_region_scores = {}
        if tamper_analysis:
            if hasattr(tamper_analysis, 'region_scores') and tamper_analysis.region_scores:
                tamper_region_scores = tamper_analysis.region_scores
            elif hasattr(tamper_analysis, 'metrics') and isinstance(tamper_analysis.metrics, dict):
                tamper_region_scores = tamper_analysis.metrics.get('region_scores', {})

        extracted_fields_list = []
        if hasattr(document, 'extracted_fields'):
            for f in document.extracted_fields.all():
                raw_val = f.field_value or ""
                # Privacy masking for Aadhaar numbers
                if f.field_name == 'aadhaar_number' and len(raw_val.replace(" ", "")) >= 12:
                    clean = raw_val.replace(" ", "")
                    masked_val = f"XXXX XXXX {clean[-4:]}"
                else:
                    masked_val = raw_val

                extracted_fields_list.append({
                    "field": f.field_name,
                    "value": masked_val,
                    "confidence": f.confidence,
                    "validation_status": "VALID" if f.confidence >= 0.70 else "LOW_CONFIDENCE"
                })

        debug_payload = {
            "verification_id": document.verification_id,
            "document_type_selected": document.document_type,
            "document_classifier_result": {
                "detected_type": quality_analysis.classified_type if quality_analysis else "unknown_document",
                "confidence": quality_analysis.classification_confidence if quality_analysis else 0.50,
                "is_type_mismatch": metrics.get("is_type_mismatch", False)
            },
            "image_quality": {
                "quality_category": metrics.get("quality_category", "ACCEPTABLE" if (quality_analysis and quality_analysis.quality_score >= 60) else "POOR"),
                "quality_score": quality_analysis.quality_score if quality_analysis else 0,
                "is_poor_quality": quality_analysis.is_poor_quality if quality_analysis else False,
                "issues": quality_analysis.issues if quality_analysis else [],
                "metrics": {
                    "blur_laplacian": metrics.get("blur_laplacian", metrics.get("blur_variance")),
                    "brightness": metrics.get("brightness", metrics.get("mean_brightness")),
                    "contrast": metrics.get("contrast", metrics.get("rms_contrast")),
                    "glare_percentage": metrics.get("glare_percentage"),
                    "resolution": metrics.get("resolution")
                }
            },
            "ocr_analysis": {
                "engine": ocr_analysis.engine_used if ocr_analysis else "EasyOCR",
                "overall_confidence": ocr_analysis.overall_confidence if ocr_analysis else 0.0,
                "extracted_fields": extracted_fields_list,
                "deskew_angle": ocr_analysis.deskew_angle if ocr_analysis else 0.0
            },
            "qr_detection": {
                "qr_detected": qr_data.get("qr_detected", False),
                "qr_decoded": qr_data.get("qr_decoded", False),
                "format_type": qr_data.get("format_type", "None"),
                "signature_status": qr_data.get("signature_status", "Official QR signature validation unavailable (Screening Prototype)"),
                "consistency_status": qr_data.get("consistency_status", qr_consistency.get("consistency_status", "NOT_CHECKED")),
                "consistency_score": qr_data.get("consistency_score", qr_consistency.get("consistency_score", 1.0)),
                "mismatched_fields": qr_data.get("mismatched_fields", qr_consistency.get("mismatched_fields", []))
            },
            "template_analysis": {
                "template_status": template_res.get("template_status", "UNKNOWN"),
                "confidence": template_res.get("confidence", 0.50),
                "checklist": template_res.get("structural_checklist", {})
            },
            "tamper_detection": {
                "tamper_score": tamper_analysis.tampering_probability if tamper_analysis else 0.0,
                "tamper_status": "HIGH" if (tamper_analysis and tamper_analysis.tampering_probability > 0.40) else ("MEDIUM" if (tamper_analysis and tamper_analysis.tampering_probability > 0.20) else "LOW"),
                "region_scores": tamper_region_scores,
                "forensic_signals": {
                    "signals": tamper_analysis.signals if tamper_analysis else [],
                    "suspicious_regions": tamper_analysis.suspicious_regions if tamper_analysis else [],
                    "metadata_findings": tamper_analysis.metadata_findings if tamper_analysis else {}
                }
            },
            "face_biometrics": {
                "document_face_detected": face_verification.document_face_detected if face_verification else False,
                "selfie_face_detected": face_verification.selfie_face_detected if face_verification else False,
                "liveness_status": (face_verification.face_quality_metrics.get("liveness", "PASSED") if (face_verification and isinstance(face_verification.face_quality_metrics, dict)) else "PASSED") if face_verification else "PASSED",
                "similarity_score": face_verification.similarity_score if face_verification else 0.0,
                "calibrated_threshold": face_verification.threshold_used if (face_verification and face_verification.threshold_used) else 0.70,
                "match_status": face_verification.match_status if face_verification else "NOT_APPLICABLE"
            },
            "identity_consistency": {
                "consistency_score": consistency_check.consistency_score if consistency_check else 100,
                "status": "CONSISTENT" if (consistency_check and consistency_check.consistency_score >= 80) else "INCONSISTENT",
                "checklist": consistency_check.checklist if consistency_check else [],
                "signals": consistency_check.signals if consistency_check else [],
                "field_matches": {
                    "name_match": consistency_check.name_match if consistency_check else True,
                    "dob_match": consistency_check.dob_match if consistency_check else True,
                    "document_number_match": consistency_check.document_number_match if consistency_check else True,
                    "gender_match": consistency_check.gender_match if consistency_check else True
                }
            },
            "risk_engine": {
                "risk_score": risk_result.get("risk_score", 0),
                "risk_category": risk_result.get("category", "LOW_RISK"),
                "explanation": risk_result.get("explanation", ""),
                "factor_contributions": risk_result.get("factors", [])
            },
            "official_verification": {
                "status": "Official UIDAI verification not performed",
                "screening_mode": "AI-based document screening"
            }
        }
        return Response(debug_payload, status=status.HTTP_200_OK)

