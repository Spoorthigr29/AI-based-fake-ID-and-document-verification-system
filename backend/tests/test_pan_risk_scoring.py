import unittest
from documents.services.aadhaar_fixes import compute_risk
from risk_engine.services.risk_calculator import DeterministicRiskCalculator

class TestPANRiskScoring(unittest.TestCase):
    """Test suite verifying balanced, explainable risk scoring for PAN documents."""

    def test_genuine_pan_document_scoring(self):
        """
        Verify the user's exact PAN scenario:
        - Document Type: PAN (Matched)
        - OCR: 62% (Moderate / Review Required)
        - QR: Unavailable
        - Document Integrity: Clean (10% anomaly / normal scan)
        - Document Photo: Detected
        - Liveness: Passed
        - Face Verification: Matched (90% similarity)
        - Identity Consistency: Consistent (100%)
        """
        checks = {
            "document_type": "PAN",
            "type_check": {
                "selected": "PAN",
                "detected": "PAN",
                "is_mismatch": False
            },
            "qr_check": {
                "qr_detected": False,
                "qr_decoded": False
            },
            "qr_consistency": {
                "status": "NOT_AVAILABLE",
                "mismatched_fields": []
            },
            "face_check": {
                "is_applicable": True,
                "document_face_detected": True,
                "match_status": "MATCH",
                "similarity_score": 0.90
            },
            "tamper_check": {
                "tampering_probability": 0.10,
                "signals": []
            },
            "ocr_confidence": 0.62,
            "identity_consistency": {
                "consistency_score": 100.0,
                "status": "CONSISTENT"
            }
        }

        doc_quality = {
            "quality_score": 85,
            "is_blurry": False
        }

        result = compute_risk(checks, doc_quality=doc_quality, is_camera_or_compressed=True)

        # 1. Category must be LOW_RISK or minimal review, definitely NOT HIGH_RISK or 50/100
        self.assertLessEqual(result["risk_score"], 29, f"Risk score {result['risk_score']} should be in LOW_RISK (<= 29)")
        self.assertEqual(result["decision"], "APPROVE")
        self.assertEqual(result["category"], "LOW_RISK")

        # 2. QR unavailable must have 0 contribution
        qr_factors = [f for f in result["factors"] if "QR" in f["name"]]
        self.assertTrue(len(qr_factors) > 0)
        self.assertEqual(qr_factors[0]["contribution"], 0.0)
        self.assertEqual(qr_factors[0]["value"], "UNAVAILABLE")

        # 3. Face verification matched must have 0 penalty
        face_factors = [f for f in result["factors"] if "Face" in f["name"]]
        self.assertTrue(len(face_factors) > 0)
        self.assertEqual(face_factors[0]["contribution"], 0.0)

        # 4. Document integrity clean must have 0 penalty
        tamper_factors = [f for f in result["factors"] if "Integrity" in f["name"]]
        self.assertTrue(len(tamper_factors) > 0)
        self.assertEqual(tamper_factors[0]["contribution"], 0.0)

        # 5. Explanation must be transparent and explain signals
        self.assertIn("low risk", result["explanation"].lower())
        self.assertNotIn("contradicts", result["explanation"].lower())

    def test_unavailable_checks_zero_penalty(self):
        """Verify unavailable face and QR checks contribute 0 penalty."""
        checks = {
            "document_type": "PAN",
            "type_check": {"selected": "PAN", "detected": "PAN", "is_mismatch": False},
            "qr_check": {"qr_detected": False, "qr_decoded": False},
            "qr_consistency": {"status": "NOT_AVAILABLE"},
            "face_check": {"is_applicable": False, "document_face_detected": False, "match_status": "NOT_APPLICABLE"},
            "tamper_check": {"tampering_probability": 0.05},
            "ocr_confidence": 0.88,
            "identity_consistency": {"consistency_score": 100.0, "status": "CONSISTENT"}
        }
        result = compute_risk(checks, doc_quality={"quality_score": 90, "is_blurry": False})
        self.assertEqual(result["risk_score"], 0)
        self.assertEqual(result["decision"], "APPROVE")

    def test_actual_tampering_and_mismatch_generates_high_risk(self):
        """Verify genuine fraud evidence (e.g. type mismatch or face mismatch) produces high risk."""
        checks = {
            "document_type": "PAN",
            "type_check": {"selected": "PAN", "detected": "AADHAAR", "is_mismatch": True},
            "qr_check": {"qr_detected": False, "qr_decoded": False},
            "face_check": {"is_applicable": True, "document_face_detected": True, "match_status": "MISMATCH", "similarity_score": 0.20},
            "tamper_check": {"tampering_probability": 0.85},
            "ocr_confidence": 0.40,
            "identity_consistency": {"consistency_score": 20.0, "status": "INCONSISTENT"}
        }
        result = compute_risk(checks, doc_quality={"quality_score": 85, "is_blurry": False})
        self.assertGreaterEqual(result["risk_score"], 60)
        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(result["category"], "HIGH_RISK")
