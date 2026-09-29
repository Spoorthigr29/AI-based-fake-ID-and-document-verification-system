"""
VerifyX AI - Deterministic & Explainable Risk Factor Evaluator
=============================================================
Calculates granular point contributions for each screening signal to produce
a transparent, auditable breakdown of overall risk.

Screening signals:
1. Document Quality (0-100)
2. OCR Confidence (0-100 or 0-1)
3. Face Similarity (0-100 or 0-1)
4. Tampering Probability (0-100 or 0-1)
5. Identity Consistency (0-100)
6. Document Classification Confidence (0-100 or 0-1)
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class RiskFactorItem:
    factor_name: str
    factor_value: str
    contribution: float
    description: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.factor_name,
            "value": self.factor_value,
            "contribution": round(self.contribution, 1),
            "description": self.description,
            "severity": self.severity
        }


class RiskFactorEvaluator:
    """
    Evaluates individual screening signals and assigns deterministic
    point contributions towards the aggregate risk score (0-100).
    """

    @staticmethod
    def _normalize_percent(val: Any, default: float = 100.0) -> float:
        """Helper to normalize values to 0.0 - 100.0 scale."""
        if val is None:
            return default
        try:
            f = float(val)
            # If value is in 0.0 - 1.0 range (e.g. 0.94), convert to percentage
            if 0.0 <= f <= 1.0 and f != 0 and f != 1:
                return f * 100.0
            if f == 1.0 and isinstance(val, float):
                return 100.0
            return max(0.0, min(100.0, f))
        except (ValueError, TypeError):
            return default

    def evaluate_all(self, signals: Dict[str, Any]) -> List[RiskFactorItem]:
        """
        Evaluate all screening signals and return a list of granular RiskFactorItem objects.
        """
        factors = []

        # 0. Document Authenticity (Deep Learning Signal)
        if 'authenticity_prediction' in signals or 'authenticity_score' in signals:
            factors.append(self.evaluate_document_authenticity(
                signals.get('authenticity_score'),
                signals.get('authenticity_prediction'),
                signals.get('authenticity_confidence'),
                context_signals=signals
            ))

        # 1. Face Similarity
        factors.append(self.evaluate_face_similarity(
            signals.get('face_similarity'),
            signals.get('face_match_status')
        ))

        # 2. Tampering Probability
        factors.append(self.evaluate_tampering_probability(
            signals.get('tampering_probability'),
            signals.get('tamper_signals')
        ))

        # 3. Identity Consistency & Sub-fields
        factors.append(self.evaluate_identity_consistency(
            signals.get('identity_consistency'),
            name_match=signals.get('name_match'),
            dob_match=signals.get('dob_match'),
            id_format_valid=signals.get('id_format_valid'),
            address_match=signals.get('address_match')
        ))

        # Optional granular field consistency factors if provided
        if 'dob_match' in signals and signals.get('dob_match') is not None:
            dob_val = bool(signals.get('dob_match'))
            if not dob_val:
                factors.append(RiskFactorItem(
                    factor_name="DOB Consistency",
                    factor_value="MISMATCH",
                    contribution=12.0,
                    description="Date of birth does not match application records or logical format",
                    severity="HIGH"
                ))

        if 'name_match' in signals and signals.get('name_match') is not None:
            name_val = bool(signals.get('name_match'))
            if not name_val:
                factors.append(RiskFactorItem(
                    factor_name="Name Consistency",
                    factor_value="MISMATCH",
                    contribution=15.0,
                    description="Applicant name differs significantly from document OCR extraction",
                    severity="HIGH"
                ))

        # 4. Document Quality
        factors.append(self.evaluate_document_quality(
            signals.get('document_quality'),
            signals.get('quality_issues')
        ))

        # 5. OCR Confidence
        factors.append(self.evaluate_ocr_confidence(
            signals.get('ocr_confidence')
        ))

        # 6. Document Classification Confidence & Type Matching
        if signals.get('is_type_mismatch'):
            factors.append(RiskFactorItem(
                factor_name="Document Type Verification",
                factor_value="MISMATCH",
                contribution=50.0,
                description=f"Selected document type ({signals.get('selected_type')}) mismatches detected document template ({signals.get('detected_type')})",
                severity="CRITICAL"
            ))
        elif 'classification_confidence' in signals or 'document_type' in signals:
            factors.append(self.evaluate_classification_confidence(
                signals.get('classification_confidence'),
                signals.get('document_type')
            ))

        # 7. QR Verification and Consistency
        if 'qr_detected' in signals:
            factors.append(self.evaluate_qr_consistency(
                signals.get('qr_detected'),
                signals.get('qr_decoded'),
                signals.get('qr_consistency_status'),
                signals.get('qr_mismatched_fields')
            ))

        # 8. Template Structural Verification
        if 'template_status' in signals:
            factors.append(self.evaluate_template_structure(
                signals.get('template_status'),
                signals.get('template_confidence'),
                signals.get('template_issues')
            ))

        return factors

    def evaluate_qr_consistency(
        self,
        qr_detected: Any,
        qr_decoded: Any,
        consistency_status: Optional[str] = None,
        mismatched_fields: Optional[List[str]] = None
    ) -> RiskFactorItem:
        if not qr_detected:
            return RiskFactorItem(
                factor_name="QR Code Verification",
                factor_value="NOT_DETECTED",
                contribution=0.0,
                description="QR code not detected on this document image.",
                severity="LOW"
            )

        if not qr_decoded:
            return RiskFactorItem(
                factor_name="QR Code Verification",
                factor_value="UNREADABLE",
                contribution=5.0,
                description="QR code detected but could not be legibly decoded due to resolution or blur.",
                severity="LOW"
            )

        if consistency_status == "INCONSISTENT":
            fields_str = ", ".join(mismatched_fields) if mismatched_fields else "Demographic mismatch"
            return RiskFactorItem(
                factor_name="QR / Document Consistency",
                factor_value="INCONSISTENT",
                contribution=45.0,
                description=f"CRITICAL DATA CONTRADICTION: Printed document contradicts digital QR record ({fields_str}).",
                severity="CRITICAL"
            )
        elif consistency_status == "CONSISTENT":
            return RiskFactorItem(
                factor_name="QR / Document Consistency",
                factor_value="MATCHED",
                contribution=0.0,
                description="Verified QR record is completely consistent with visible OCR document text.",
                severity="LOW"
            )
        else:
            return RiskFactorItem(
                factor_name="QR Code Verification",
                factor_value="DECODED",
                contribution=0.0,
                description="QR code successfully decoded and structure parsed.",
                severity="LOW"
            )

    def evaluate_template_structure(
        self,
        template_status: Optional[str] = None,
        confidence: Optional[float] = None,
        issues: Optional[List[str]] = None
    ) -> RiskFactorItem:
        status = template_status or "UNKNOWN"
        conf = float(confidence or 0.8)

        if status == "CONSISTENT":
            return RiskFactorItem(
                factor_name="Template Structural Verification",
                factor_value="CONSISTENT",
                contribution=0.0,
                description="Document structure, emblems, and visual layout are consistent with standard template.",
                severity="LOW"
            )
        elif status == "SUSPICIOUS":
            return RiskFactorItem(
                factor_name="Template Structural Verification",
                factor_value="SUSPICIOUS",
                contribution=15.0,
                description=f"Document layout deviates from standard template ({'; '.join(issues or ['Layout anomaly'])}).",
                severity="MEDIUM"
            )
        else:
            return RiskFactorItem(
                factor_name="Template Structural Verification",
                factor_value="UNKNOWN",
                contribution=0.0,
                description="Template visual layout evaluated with baseline confidence.",
                severity="LOW"
            )

    def evaluate_document_authenticity(
        self,
        authenticity_score: Any = None,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        context_signals: Optional[Dict[str, Any]] = None
    ) -> RiskFactorItem:
        if prediction is None and authenticity_score is None:
            return RiskFactorItem(
                factor_name="Document Authenticity",
                factor_value="NOT_EVALUATED",
                contribution=0.0,
                description="Deep learning authenticity model not evaluated.",
                severity="LOW"
            )

        is_fake = (prediction == "FAKE") or (authenticity_score is not None and float(authenticity_score) < 50.0)
        conf = float(confidence) if confidence is not None else 85.0

        if is_fake:
            # Check if multi-modal biometric & consistency evidence is overwhelmingly positive
            sig = context_signals or {}
            face_match = sig.get('face_match_status') == 'MATCH' or (sig.get('face_similarity', 0.0) or 0.0) >= 0.70
            cons_score = float(sig.get('identity_consistency', 100.0) or 100.0)
            ocr_conf = float(sig.get('ocr_confidence', 0.0) or 0.0)
            
            # If all other verification layers are positive (genuine physical photo of ID with live person)
            if face_match and cons_score >= 80.0 and ocr_conf >= 0.50:
                contrib = 10.0
                sev = "MEDIUM"
                desc = f"Visual classifier noted camera/photo compression variance ({conf:.1f}% flag); other primary signals verified."
            else:
                contrib = 38.0 if conf >= 85.0 else (28.0 if conf >= 65.0 else 20.0)
                sev = "CRITICAL" if conf >= 80 else "HIGH"
                desc = f"Deep learning authenticity model flagged synthetic/forged patterns with {conf:.1f}% confidence."

            return RiskFactorItem(
                factor_name="Document Authenticity",
                factor_value=f"FLAGGED ({conf:.1f}%)" if sev == "MEDIUM" else f"FAKE ({conf:.1f}%)",
                contribution=contrib,
                description=desc,
                severity=sev
            )
        else:
            return RiskFactorItem(
                factor_name="Document Authenticity",
                factor_value=f"REAL ({conf:.1f}%)",
                contribution=0.0,
                description=f"Deep learning model verified authentic document structure and security textures ({conf:.1f}% confidence).",
                severity="LOW"
            )

    def evaluate_face_similarity(self, similarity: Any, match_status: Optional[str] = None) -> RiskFactorItem:
        if match_status in ['NOT_APPLICABLE', 'NOT_PERFORMED']:
            return RiskFactorItem(
                factor_name="Face Verification",
                factor_value="NOT_APPLICABLE",
                contribution=0.0,
                description="Face comparison is not applicable because the selected document template does not contain a photograph.",
                severity="LOW"
            )

        val = self._normalize_percent(similarity, default=90.0)
        
        if match_status in ['MISMATCH', 'NO_FACE', 'MULTI_FACE'] or val < 40.0:
            return RiskFactorItem(
                factor_name="Face Similarity",
                factor_value=f"{val:.0f}%",
                contribution=35.0,
                description=f"Significant biometric discrepancy or invalid face detection ({match_status or 'Low Similarity'}).",
                severity="CRITICAL"
            )
        elif val < 60.0 or match_status == 'POOR_QUALITY':
            return RiskFactorItem(
                factor_name="Face Similarity",
                factor_value=f"{val:.0f}%",
                contribution=22.0,
                description="Moderate biometric match uncertainty below standard threshold.",
                severity="HIGH"
            )
        elif val < 75.0 or match_status == 'MANUAL_REVIEW':
            return RiskFactorItem(
                factor_name="Face Similarity",
                factor_value=f"{val:.0f}%",
                contribution=10.0,
                description="Borderline face similarity score requiring secondary review.",
                severity="MEDIUM"
            )
        elif val < 90.0:
            return RiskFactorItem(
                factor_name="Face Similarity",
                factor_value=f"{val:.0f}%",
                contribution=4.0,
                description="Acceptable face match with minor facial angle/lighting variation.",
                severity="LOW"
            )
        else:
            return RiskFactorItem(
                factor_name="Face Similarity",
                factor_value=f"{val:.0f}%",
                contribution=1.0 if val < 95 else 0.0,
                description="High-confidence biometric verification matched against selfie.",
                severity="LOW"
            )

    def evaluate_tampering_probability(self, tampering: Any, signals: Optional[list] = None) -> RiskFactorItem:
        # Tampering is high risk when value is HIGH (e.g. 80% tampering is bad)
        val = self._normalize_percent(tampering, default=5.0)

        if val >= 60.0:
            return RiskFactorItem(
                factor_name="Tampering Probability",
                factor_value=f"{val:.0f}%",
                contribution=40.0,
                description="Critical forensic anomalies detected (high ELA variance / copy-move artifacts).",
                severity="CRITICAL"
            )
        elif val >= 38.0:
            return RiskFactorItem(
                factor_name="Tampering Probability",
                factor_value=f"{val:.0f}%",
                contribution=22.0,
                description="Moderate digital manipulation risk flagged in image forensic analysis.",
                severity="HIGH"
            )
        elif val >= 22.0:
            return RiskFactorItem(
                factor_name="Tampering Probability",
                factor_value=f"{val:.0f}%",
                contribution=12.0,
                description="Moderate/uncertain forensic evidence requiring visual officer inspection.",
                severity="MEDIUM"
            )
        elif val >= 12.0:
            return RiskFactorItem(
                factor_name="Tampering Probability",
                factor_value=f"{val:.0f}%",
                contribution=3.0,
                description="Minor benign compression or natural edge texture variations detected.",
                severity="LOW"
            )
        else:
            return RiskFactorItem(
                factor_name="Tampering Probability",
                factor_value=f"{val:.0f}%",
                contribution=0.0,
                description="Clean digital forensic signature with little to no detectable manipulation.",
                severity="LOW"
            )

    def evaluate_identity_consistency(
        self,
        consistency: Any,
        name_match: Optional[Any] = None,
        dob_match: Optional[Any] = None,
        id_format_valid: Optional[Any] = None,
        address_match: Optional[Any] = None
    ) -> RiskFactorItem:
        val = self._normalize_percent(consistency, default=95.0)

        # Count boolean failures if provided
        failures = 0
        if name_match is not None and not bool(name_match):
            failures += 1
        if dob_match is not None and not bool(dob_match):
            failures += 1
        if id_format_valid is not None and not bool(id_format_valid):
            failures += 1
        if address_match is not None and not bool(address_match):
            failures += 0.5

        if val < 40.0 or failures >= 3:
            return RiskFactorItem(
                factor_name="Identity Consistency",
                factor_value=f"{val:.0f}%",
                contribution=30.0,
                description="Major cross-field discrepancy across applicant data and document fields.",
                severity="CRITICAL"
            )
        elif val < 70.0 or failures >= 2:
            return RiskFactorItem(
                factor_name="Identity Consistency",
                factor_value=f"{val:.0f}%",
                contribution=18.0,
                description="Multiple cross-field validation discrepancies detected.",
                severity="HIGH"
            )
        elif val < 85.0 or failures >= 1:
            return RiskFactorItem(
                factor_name="Identity Consistency",
                factor_value=f"{val:.0f}%",
                contribution=6.0,
                description="Minor field mismatch (e.g. address variation or spelling noise).",
                severity="MEDIUM"
            )
        else:
            return RiskFactorItem(
                factor_name="Identity Consistency",
                factor_value=f"{val:.0f}%",
                contribution=0.0,
                description="Cross-field logical verification consistent with intake records.",
                severity="LOW"
            )

    def evaluate_document_quality(self, quality: Any, issues: Optional[list] = None) -> RiskFactorItem:
        val = self._normalize_percent(quality, default=90.0)

        if val < 40.0:
            return RiskFactorItem(
                factor_name="Document Quality",
                factor_value=f"{val:.0f}%",
                contribution=20.0,
                description="Severely degraded image (extreme blur, glare, or sub-standard resolution).",
                severity="HIGH"
            )
        elif val < 65.0:
            return RiskFactorItem(
                factor_name="Document Quality",
                factor_value=f"{val:.0f}%",
                contribution=10.0,
                description="Marginal image quality with readability defects.",
                severity="MEDIUM"
            )
        elif val < 85.0:
            return RiskFactorItem(
                factor_name="Document Quality",
                factor_value=f"{val:.0f}%",
                contribution=3.0,
                description="Acceptable document scan quality with minor noise.",
                severity="LOW"
            )
        else:
            return RiskFactorItem(
                factor_name="Document Quality",
                factor_value=f"{val:.0f}%",
                contribution=0.0,
                description="Clear, high-resolution document image meeting quality standards.",
                severity="LOW"
            )

    def evaluate_ocr_confidence(self, ocr_conf: Any) -> RiskFactorItem:
        val = self._normalize_percent(ocr_conf, default=95.0)

        if val < 45.0:
            return RiskFactorItem(
                factor_name="OCR Confidence",
                factor_value=f"{val:.0f}%",
                contribution=15.0,
                description="Low OCR character confidence indicating illegible or manipulated text.",
                severity="HIGH"
            )
        elif val < 60.0:
            return RiskFactorItem(
                factor_name="OCR Confidence",
                factor_value=f"{val:.0f}%",
                contribution=5.0,
                description="Moderate optical recognition confidence; some tokens uncertain.",
                severity="MEDIUM"
            )
        elif val < 85.0:
            return RiskFactorItem(
                factor_name="OCR Confidence",
                factor_value=f"{val:.0f}%",
                contribution=2.0,
                description="Good OCR transcription confidence across all key fields.",
                severity="LOW"
            )
        else:
            return RiskFactorItem(
                factor_name="OCR Confidence",
                factor_value=f"{val:.0f}%",
                contribution=0.0,
                description="High optical recognition confidence across all key fields.",
                severity="LOW"
            )

    def evaluate_classification_confidence(self, class_conf: Any, doc_type: Optional[str] = None) -> RiskFactorItem:
        val = self._normalize_percent(class_conf, default=90.0)
        is_unknown = doc_type in ['unknown_document', 'OTHER', None]

        if is_unknown or val < 40.0:
            return RiskFactorItem(
                factor_name="Classification Confidence",
                factor_value=f"{val:.0f}%",
                contribution=15.0,
                description=f"Unrecognized document type layout or low template match confidence ({doc_type or 'Unknown'}).",
                severity="MEDIUM"
            )
        elif val < 70.0:
            return RiskFactorItem(
                factor_name="Classification Confidence",
                factor_value=f"{val:.0f}%",
                contribution=6.0,
                description="Document template classified with moderate confidence.",
                severity="LOW"
            )
        else:
            return RiskFactorItem(
                factor_name="Classification Confidence",
                factor_value=f"{val:.0f}%",
                contribution=0.0,
                description="Document layout reliably matched against recognized synthetic template.",
                severity="LOW"
            )
