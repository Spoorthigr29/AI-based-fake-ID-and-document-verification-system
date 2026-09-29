from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from documents.models import Document
from audit.models import AuditLog
from .models import OCRAnalysis, ExtractedField
from .services.ocr_service import OCREngineService

class OCRProcessAPIView(APIView):
    """
    POST /api/ocr/process/<verification_id>/
    Triggers the OCR & Field Extraction pipeline on the uploaded document.
    Stores raw text and structured fields into the database, returning extracted attributes with confidence scores.
    """
    def post(self, request, verification_id, *args, **kwargs):
        try:
            document = Document.objects.get(verification_id=verification_id)
        except Document.DoesNotExist:
            return Response({
                "success": False,
                "error": f"Verification case '{verification_id}' not found."
            }, status=status.HTTP_404_NOT_FOUND)

        if not document.original_file:
            return Response({
                "success": False,
                "error": "No identity document file attached to this verification record."
            }, status=status.HTTP_400_BAD_REQUEST)

        # Execute OCR Pipeline
        try:
            file_path = document.original_file.path
            result = OCREngineService.process_document(file_path)
        except Exception as e:
            # Try reading directly from file storage if path cannot be accessed directly
            try:
                document.original_file.seek(0)
                result = OCREngineService.process_document(document.original_file)
            except Exception as inner_e:
                return Response({
                    "success": False,
                    "error": f"OCR extraction failed: {str(inner_e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        fields = result.get("fields", {})
        raw_text = result.get("raw_text", "")
        overall_confidence = result.get("overall_confidence", 0.0)
        engine_used = result.get("engine_used", "None")
        quality_metrics = result.get("quality_metrics", {})
        deskew_angle = result.get("deskew_angle", 0.0)

        # Store / Update OCRAnalysis in database
        ocr_record, _ = OCRAnalysis.objects.update_or_create(
            document=document,
            defaults={
                "raw_text": raw_text,
                "overall_confidence": overall_confidence,
                "engine_used": engine_used,
                "quality_metrics": quality_metrics,
                "deskew_angle": deskew_angle,
            }
        )

        # Clear previously extracted fields for idempotency and recreate
        ExtractedField.objects.filter(document=document).delete()
        for field_name, field_data in fields.items():
            if isinstance(field_data, dict):
                val = field_data.get("value", "")
                conf = field_data.get("confidence", 0.0)
            else:
                val = str(field_data)
                conf = overall_confidence

            ExtractedField.objects.create(
                document=document,
                field_name=field_name,
                field_value=str(val),
                confidence=float(conf)
            )

        # Update Document processing status
        document.processing_status = Document.STATUS_PROCESSING
        document.save(update_fields=['processing_status'])

        # Audit Log
        AuditLog.objects.create(
            user=request.user if request.user.is_authenticated else None,
            action=AuditLog.ACTION_AI_SCAN,
            document=document,
            metadata={
                "pipeline": "OCR_EXTRACTION",
                "engine": engine_used,
                "fields_extracted_count": len(fields),
                "overall_confidence": overall_confidence,
                "quality_score": quality_metrics.get("quality_score", 0.0)
            }
        )

        return Response({
            "success": True,
            "verification_id": document.verification_id,
            "fields": fields
        }, status=status.HTTP_200_OK)
