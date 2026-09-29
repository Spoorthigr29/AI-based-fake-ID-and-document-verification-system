"""
VerifyX AI - Deterministic & Explainable Risk Calculator
========================================================
Central Risk Scoring Engine that consumes screening signals from:
1. Document quality
2. OCR confidence
3. Face similarity
4. Tampering probability
5. Identity consistency
6. Document classification confidence

Generates:
- Risk Score: 0 - 100
- Risk Category:
    * 0 - 30: LOW_RISK
    * 31 - 70: MANUAL_REVIEW
    * 71 - 100: HIGH_RISK
- Granular Risk Factors list
- Transparent natural language explanation

Architecture is designed with a swappable calculator interface
to allow replacement or combination with ML models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from django.db import transaction

from .risk_factors import RiskFactorEvaluator, RiskFactorItem
from .explanation_generator import ExplanationGenerator


class BaseRiskCalculator(ABC):
    """Abstract base class interface for VerifyX risk calculators."""

    @abstractmethod
    def calculate_risk(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Compute risk score, category, factors, and explanation from screening signals."""
        pass


class DeterministicRiskCalculator(BaseRiskCalculator):
    """
    Deterministic, rule-based, and explainable risk calculation engine.
    Computes an auditable risk score (0-100) and stores granular factor contributions.
    """

    def __init__(self):
        self.evaluator = RiskFactorEvaluator()
        self.explanation_gen = ExplanationGenerator()

    @staticmethod
    def determine_category(risk_score: int) -> str:
        """
        Classify the risk score into standard VerifyX risk categories.
        0 - 29: LOW_RISK
        30 - 59: MANUAL_REVIEW
        60 - 100: HIGH_RISK
        """
        if risk_score <= 29:
            return "LOW_RISK"
        elif risk_score <= 59:
            return "MANUAL_REVIEW"
        else:
            return "HIGH_RISK"

    def calculate_risk(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk score from raw or normalized screening signals using dynamic weight re-normalization.
        """
        from documents.services.aadhaar_fixes import compute_risk

        checks = {
            "document_type": signals.get("document_type") or signals.get("selected_type") or signals.get("detected_type"),
            "qr_check": {
                "qr_detected": signals.get("qr_detected"),
                "qr_decoded": signals.get("qr_decoded"),
            },
            "qr_consistency": {
                "status": signals.get("qr_consistency_status"),
                "mismatched_fields": signals.get("qr_mismatched_fields", [])
            },
            "face_check": {
                "is_applicable": signals.get("face_match_status") not in ["NOT_APPLICABLE", "NOT_PERFORMED"],
                "document_face_detected": signals.get("document_face_detected", True) if signals.get("face_match_status") != "NOT_APPLICABLE" else False,
                "match_status": signals.get("face_match_status"),
                "similarity_score": signals.get("face_similarity")
            },
            "tamper_check": {
                "tampering_probability": signals.get("tampering_probability", 0.05),
                "signals": signals.get("tamper_signals", [])
            },
            "type_check": {
                "selected": signals.get("selected_type"),
                "detected": signals.get("detected_type"),
                "is_mismatch": signals.get("is_type_mismatch", False)
            },
            "ocr_confidence": signals.get("ocr_confidence", 0.90),
            "identity_consistency": {
                "consistency_score": signals.get("identity_consistency", 100.0),
                "status": "CONSISTENT" if signals.get("identity_consistency", 100.0) >= 70 else "MANUAL_REVIEW"
            }
        }
        doc_quality = {
            "quality_score": signals.get("document_quality", 85),
            "is_blurry": any("blur" in str(i).lower() for i in signals.get("quality_issues", []))
        }

        # Run compute_risk with compression tolerance
        risk_res = compute_risk(checks, doc_quality=doc_quality, is_camera_or_compressed=True)

        return {
            "risk_score": risk_res["risk_score"],
            "decision": risk_res["decision"],
            "category": risk_res["category"],
            "factors": risk_res["factors"],
            "explanation": risk_res["explanation"],
            "rescan_message": risk_res.get("rescan_message"),
            "signals_evaluated": len(risk_res["factors"])
        }

    def evaluate_for_document(self, document_or_id: Any) -> Dict[str, Any]:
        """
        Extract screening signals from related models for a given Document or verification_id,
        calculate the risk score, and persist VerificationResult and RiskFactor models.
        """
        from documents.models import Document
        from identity_verification.models import VerificationResult
        from risk_engine.models import RiskFactor

        # Resolve Document instance
        if isinstance(document_or_id, Document):
            document = document_or_id
        else:
            document = Document.objects.filter(verification_id=str(document_or_id)).first()
            if not document:
                # Try by pk if not found
                document = Document.objects.filter(pk=document_or_id).first()
            if not document:
                raise ValueError(f"Document with identifier '{document_or_id}' not found.")

        # Harvest signals across all VerifyX modules
        signals = {}

        # 1. Document Quality & Authenticity
        quality = getattr(document, 'quality_analysis', None)
        if quality:
            signals['document_quality'] = quality.quality_score
            signals['quality_issues'] = quality.issues
            signals['classification_confidence'] = quality.classification_confidence
            signals['document_type'] = quality.classified_type
            if isinstance(quality.metrics, dict):
                if 'authenticity' in quality.metrics:
                    auth_data = quality.metrics['authenticity']
                    signals['authenticity_prediction'] = auth_data.get('authenticity_prediction')
                    signals['authenticity_confidence'] = auth_data.get('confidence_score')
                    signals['authenticity_score'] = auth_data.get('authenticity_score')
                if 'type_mismatch' in quality.metrics:
                    signals['is_type_mismatch'] = quality.metrics.get('type_mismatch')
                    signals['selected_type'] = document.document_type
                    signals['detected_type'] = quality.classified_type
                if 'qr' in quality.metrics:
                    qr_meta = quality.metrics['qr']
                    signals['qr_detected'] = qr_meta.get('qr_detected')
                    signals['qr_decoded'] = qr_meta.get('qr_decoded')
                    signals['qr_consistency_status'] = qr_meta.get('consistency_status')
                    signals['qr_mismatched_fields'] = qr_meta.get('mismatched_fields', [])
                if 'template_analysis' in quality.metrics:
                    tmpl_meta = quality.metrics['template_analysis']
                    signals['template_status'] = tmpl_meta.get('template_status')
                    signals['template_confidence'] = tmpl_meta.get('confidence')
                    signals['template_issues'] = tmpl_meta.get('issues', [])

        # 2. OCR Analysis
        ocr = getattr(document, 'ocr_analysis', None)
        if ocr:
            signals['ocr_confidence'] = ocr.overall_confidence

        # 3. Face Verification
        face = getattr(document, 'face_verification', None)
        if face:
            signals['face_similarity'] = face.similarity_score
            signals['face_match_status'] = face.match_status

        # 4. Tamper Detection
        tamper = getattr(document, 'tamper_analysis', None)
        if tamper:
            signals['tampering_probability'] = tamper.tampering_probability
            signals['tamper_signals'] = tamper.signals

        # 5. Identity Consistency
        consistency = getattr(document, 'consistency_check', None)
        if consistency:
            signals['identity_consistency'] = consistency.consistency_score
            signals['name_match'] = consistency.name_match
            signals['dob_match'] = consistency.dob_match
            signals['id_format_valid'] = consistency.document_number_match
            signals['address_match'] = consistency.address_match

        # Run calculation
        result = self.calculate_risk(signals)

        # Map internal category to VerificationResult model choices
        category_mapping = {
            "LOW_RISK": VerificationResult.CATEGORY_LOW_RISK,
            "MANUAL_REVIEW": VerificationResult.CATEGORY_MODERATE_RISK,
            "HIGH_RISK": VerificationResult.CATEGORY_HIGH_RISK
        }
        db_category = category_mapping.get(result['category'], VerificationResult.CATEGORY_LOW_RISK)

        # Normalize face score for database record (100.0 if not applicable)
        raw_face = signals.get('face_similarity')
        if raw_face is None:
            db_face_score = 100.0
        elif raw_face <= 1.0:
            db_face_score = raw_face * 100.0
        else:
            db_face_score = float(raw_face)

        # Persist VerificationResult & RiskFactors atomically
        with transaction.atomic():
            v_result, _ = VerificationResult.objects.update_or_create(
                document=document,
                defaults={
                    'overall_risk_score': result['risk_score'],
                    'risk_category': db_category,
                    'ocr_score': (signals.get('ocr_confidence', 0.0) * 100) if signals.get('ocr_confidence', 0.0) <= 1.0 else signals.get('ocr_confidence', 0.0),
                    'face_score': db_face_score,
                    'tamper_score': max(0.0, 100.0 - ((signals.get('tampering_probability', 0.0) * 100) if signals.get('tampering_probability', 0.0) <= 1.0 else signals.get('tampering_probability', 0.0))),
                    'consistency_score': signals.get('identity_consistency', 0.0),
                    'explanation': result['explanation']
                }
            )

            # Recreate granular RiskFactor records
            RiskFactor.objects.filter(verification_result=v_result).delete()
            factor_objects = [
                RiskFactor(
                    verification_result=v_result,
                    factor_name=f['name'],
                    factor_value=str(f['value']),
                    contribution=f['contribution'],
                    description=f['description']
                )
                for f in result['factors']
            ]
            RiskFactor.objects.bulk_create(factor_objects)

        result['verification_id'] = document.verification_id
        return result
