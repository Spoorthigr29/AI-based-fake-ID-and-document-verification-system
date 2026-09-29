from typing import Dict, Any, Optional, List
from documents.models import Document
from ocr_engine.models import OCRAnalysis, ExtractedField
from identity_verification.models import IdentityConsistencyCheck, VerificationResult
from .field_comparator import FieldComparator, FieldComparisonResult

class IdentityConsistencyEngine:
    """
    Cross-Source and Cross-Document Field Consistency Engine for VerifyX AI.
    Validates name, DOB, ID number, address, and gender fields, generating
    explainable checklist items and composite consistency scores (0-100).
    """

    def __init__(self, comparator: Optional[FieldComparator] = None):
        self.comparator = comparator or FieldComparator()

    def process_consistency(
        self,
        document: Document,
        applicant_claims: Optional[Dict[str, Any]] = None,
        ocr_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute field consistency evaluation comparing OCR extracted fields with applicant claims / reference fields.
        
        Returns:
            Dict strictly matching the requested API schema:
            {
                "name_match": bool,
                "dob_match": bool,
                "document_number_match": bool,
                "address_match": bool,
                "consistency_score": int
            }
        """
        # 1. Gather OCR Extracted Fields from Document (or use pre-extracted from pipeline)
        if ocr_fields is not None:
            clean_ocr = {}
            for k, v in ocr_fields.items():
                val = v.get("value", "") if isinstance(v, dict) else str(v)
                clean_ocr[k.lower()] = val
            ocr_fields = clean_ocr
        else:
            ocr_fields = self._extract_ocr_dict(document)

        # 2. Gather Applicant Claimed / Reference Data
        claims = applicant_claims or self._infer_claims(document, ocr_fields)

        # 3. Compare Applicable Fields Dynamically
        field_evaluations = []
        
        # Name comparison
        name_a = ocr_fields.get("name")
        name_b = claims.get("name")
        if name_a or name_b:
            comp_name = self.comparator.compare_names(name_a, name_b)
            field_evaluations.append(("name", comp_name, 35.0))
        else:
            comp_name = FieldComparisonResult(
                is_match=True, similarity_score=1.0, field_name="name",
                val_a=None, val_b=None, normalized_a=None, normalized_b=None,
                note="Name not present for comparison (N/A)"
            )

        # DOB comparison
        dob_a = ocr_fields.get("date_of_birth") or ocr_fields.get("dob")
        dob_b = claims.get("date_of_birth") or claims.get("dob")
        if dob_a or dob_b:
            comp_dob = self.comparator.compare_dobs(dob_a, dob_b)
            field_evaluations.append(("dob", comp_dob, 25.0))
        else:
            comp_dob = FieldComparisonResult(
                is_match=True, similarity_score=1.0, field_name="dob",
                val_a=None, val_b=None, normalized_a=None, normalized_b=None,
                note="DOB not present for comparison (N/A)"
            )

        # Document Number comparison
        id_a = ocr_fields.get("document_number") or ocr_fields.get("id_number") or ocr_fields.get("pan_number") or ocr_fields.get("aadhaar_number")
        id_b = claims.get("document_number") or claims.get("id_number") or claims.get("pan_number") or claims.get("aadhaar_number")
        if id_a or id_b:
            comp_id = self.comparator.compare_document_numbers(id_a, id_b)
            field_evaluations.append(("id", comp_id, 25.0))
        else:
            comp_id = FieldComparisonResult(
                is_match=True, similarity_score=1.0, field_name="document_number",
                val_a=None, val_b=None, normalized_a=None, normalized_b=None,
                note="Document ID not present for comparison (N/A)"
            )

        # Address comparison (only if present in at least one record)
        addr_a = ocr_fields.get("address")
        addr_b = claims.get("address")
        if addr_a or addr_b:
            comp_addr = self.comparator.compare_addresses(addr_a, addr_b)
            field_evaluations.append(("address", comp_addr, 15.0))
        else:
            comp_addr = FieldComparisonResult(
                is_match=True, similarity_score=1.0, field_name="address",
                val_a=None, val_b=None, normalized_a=None, normalized_b=None,
                note="Address not present on this document type (N/A)"
            )

        # Gender comparison
        comp_gen = self.comparator.compare_genders(
            ocr_fields.get("gender"), claims.get("gender")
        )

        # 4. Calculate Composite Consistency Score (0 - 100) dynamically
        checklist = []
        signals = []
        actual_mismatches = []

        if not field_evaluations:
            # Low OCR confidence or no fields extracted
            consistency_score = 85
            status = IdentityConsistencyCheck.STATUS_MANUAL_REVIEW
            checklist.append("ℹ Identity consistency deferred — OCR extraction incomplete")
            signals.append("OCR confidence is insufficient to reliably compare the extracted identity fields.")
        else:
            total_eval_weight = sum(w for _, _, w in field_evaluations)
            weighted_points = sum(res.similarity_score * w for _, res, w in field_evaluations)
            consistency_score = round(max(0, min(100, (weighted_points / total_eval_weight) * 100.0)))

            for f_name, comp_res, _ in field_evaluations:
                if comp_res.is_match:
                    checklist.append(f"✓ {f_name.replace('_', ' ').title()} consistent")
                else:
                    # Check if it is an actual contradiction or missing field
                    if comp_res.val_a and comp_res.val_b and comp_res.similarity_score < 0.60:
                        actual_mismatches.append(f_name)
                        checklist.append(f"✕ {f_name.replace('_', ' ').title()} mismatch")
                        signals.append(f"Mismatch in {f_name}: '{comp_res.val_a}' vs '{comp_res.val_b}'")
                    else:
                        checklist.append(f"⚠ {f_name.replace('_', ' ').title()} requires review")
                        signals.append(comp_res.note)

            if len(actual_mismatches) > 0:
                status = IdentityConsistencyCheck.STATUS_MANUAL_REVIEW
                document.processing_status = Document.STATUS_MANUAL_REVIEW
                document.save(update_fields=['processing_status'])
            elif consistency_score >= 70:
                status = IdentityConsistencyCheck.STATUS_CONSISTENT
            else:
                status = IdentityConsistencyCheck.STATUS_MANUAL_REVIEW

        field_details = {
            "name": {
                "extracted": comp_name.val_a,
                "claimed": comp_name.val_b,
                "normalized_a": comp_name.normalized_a,
                "normalized_b": comp_name.normalized_b,
                "match": comp_name.is_match,
                "score": comp_name.similarity_score,
                "note": comp_name.note
            },
            "date_of_birth": {
                "extracted": comp_dob.val_a,
                "claimed": comp_dob.val_b,
                "normalized_a": comp_dob.normalized_a,
                "normalized_b": comp_dob.normalized_b,
                "match": comp_dob.is_match,
                "score": comp_dob.similarity_score,
                "note": comp_dob.note
            },
            "document_number": {
                "extracted": comp_id.val_a,
                "claimed": comp_id.val_b,
                "normalized_a": comp_id.normalized_a,
                "normalized_b": comp_id.normalized_b,
                "match": comp_id.is_match,
                "score": comp_id.similarity_score,
                "note": comp_id.note
            },
            "address": {
                "extracted": comp_addr.val_a,
                "claimed": comp_addr.val_b,
                "normalized_a": comp_addr.normalized_a,
                "normalized_b": comp_addr.normalized_b,
                "match": comp_addr.is_match,
                "score": comp_addr.similarity_score,
                "note": comp_addr.note
            },
            "gender": {
                "extracted": comp_gen.val_a,
                "claimed": comp_gen.val_b,
                "match": comp_gen.is_match,
                "note": comp_gen.note
            }
        }

        # 6. Persist Record to Database
        IdentityConsistencyCheck.objects.update_or_create(
            document=document,
            defaults={
                "name_match": comp_name.is_match,
                "dob_match": comp_dob.is_match,
                "document_number_match": comp_id.is_match,
                "address_match": comp_addr.is_match,
                "gender_match": comp_gen.is_match,
                "consistency_score": consistency_score,
                "status": status,
                "field_details": field_details,
                "signals": signals,
                "checklist": checklist,
            }
        )

        return {
            "name_match": comp_name.is_match,
            "dob_match": comp_dob.is_match,
            "document_number_match": comp_id.is_match,
            "address_match": comp_addr.is_match,
            "consistency_score": consistency_score,
        }

    def _extract_ocr_dict(self, document: Document) -> Dict[str, str]:
        """Extract structured OCR field values from document OCR models or raw text."""
        fields = {}
        for ef in document.extracted_fields.all():
            fields[ef.field_name.lower()] = ef.field_value

        if not fields and document.original_file:
            # Fallback to field extractor on OCR text if not already run
            try:
                from ocr_engine.services.ocr_service import OCREngineService
                service = OCREngineService()
                res = service.process_document(document)
                for f_name, f_data in res.get("fields", {}).items():
                    fields[f_name.lower()] = f_data.get("value", "")
            except Exception:
                pass

        return fields

    def _infer_claims(self, document: Document, ocr_fields: Dict[str, str]) -> Dict[str, str]:
        """
        Infer reference claims from user profile, metadata, or baseline OCR values.
        Staff/reviewer usernames are never treated as applicant reference claims.
        """
        claims = {}
        # Only use uploaded_by if it represents a non-staff applicant with valid profile name
        if document.uploaded_by and not document.uploaded_by.is_staff and not document.uploaded_by.is_superuser:
            full_name = f"{document.uploaded_by.first_name} {document.uploaded_by.last_name}".strip()
            if full_name and len(full_name) > 2:
                claims["name"] = full_name

        # Default to document OCR extractions for internal format and structure consistency validation
        for k, v in ocr_fields.items():
            if k not in claims and v:
                claims[k] = v

        return claims
