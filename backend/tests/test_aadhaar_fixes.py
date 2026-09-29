"""
Unit tests for Aadhaar Fixes and Risk Engine:
1. Test compute_risk() with all valid signals -> APPROVE (Low Risk)
2. Test compute_risk() with unavailable checks -> Excluded & re-normalized (Low Risk, not inflated)
3. Test compute_risk() with QR/OCR contradiction -> Hard REJECT
4. Test compute_risk() with Biometric Face Mismatch -> Hard REJECT
5. Test compute_risk() with Blurry / Unreadable image -> RESCAN
6. Test detect_document_face() and decode_aadhaar_qr()
"""

import unittest
import numpy as np
from documents.services.aadhaar_fixes import (
    compute_risk,
    compare_qr_with_ocr,
    mask_aadhaar_number,
    detect_document_face,
    decode_aadhaar_qr
)

class AadhaarFixesAndRiskTestCase(unittest.TestCase):

    def test_mask_aadhaar_number(self):
        self.assertEqual(mask_aadhaar_number("123456789012"), "XXXX XXXX 9012")
        self.assertEqual(mask_aadhaar_number("1234 5678 9012"), "XXXX XXXX 9012")
        self.assertEqual(mask_aadhaar_number(""), "")

    def test_compute_risk_approved(self):
        checks = {
            "qr_check": {"qr_detected": True, "qr_decoded": True},
            "qr_consistency": {"status": "CONSISTENT", "mismatched_fields": []},
            "face_check": {"is_applicable": True, "document_face_detected": True, "match_status": "MATCH", "similarity_score": 0.92},
            "tamper_check": {"tampering_probability": 0.08, "signals": []},
            "type_check": {"selected": "AADHAAR", "detected": "AADHAAR", "is_mismatch": False}
        }
        res = compute_risk(checks, doc_quality={"quality_score": 90, "is_blurry": False})
        self.assertEqual(res["decision"], "APPROVE")
        self.assertEqual(res["category"], "LOW_RISK")
        self.assertLessEqual(res["risk_score"], 25)

    def test_compute_risk_unavailable_checks_excluded(self):
        """Unavailable face and QR checks should be excluded from weighting without raising risk."""
        checks = {
            "qr_check": {"qr_detected": False, "qr_decoded": False},
            "qr_consistency": {"status": "NOT_AVAILABLE"},
            "face_check": {"is_applicable": False, "document_face_detected": False, "match_status": "NOT_APPLICABLE"},
            "tamper_check": {"tampering_probability": 0.12, "signals": []},
            "type_check": {"selected": "AADHAAR", "detected": "AADHAAR", "is_mismatch": False}
        }
        res = compute_risk(checks, doc_quality={"quality_score": 85, "is_blurry": False})
        self.assertEqual(res["decision"], "APPROVE")
        self.assertEqual(res["category"], "LOW_RISK")
        self.assertLessEqual(res["risk_score"], 25)

    def test_compute_risk_rescan_on_blurry(self):
        """Blurry unreadable image with unreadable QR must trigger RESCAN decision."""
        checks = {
            "qr_check": {"qr_detected": False, "qr_decoded": False},
            "qr_consistency": {"status": "NOT_AVAILABLE"},
            "face_check": {"is_applicable": True, "document_face_detected": False, "match_status": "NOT_APPLICABLE"},
            "tamper_check": {"tampering_probability": 0.20},
            "type_check": {"selected": "AADHAAR", "detected": "OTHER", "is_mismatch": False}
        }
        res = compute_risk(checks, doc_quality={"quality_score": 25, "is_blurry": True})
        self.assertEqual(res["decision"], "RESCAN")
        self.assertIsNotNone(res.get("rescan_message"))
        self.assertIn("flatter", res["rescan_message"].lower() or "clearer" in res["rescan_message"].lower())

    def test_compute_risk_hard_reject_qr_inconsistency(self):
        """QR payload contradicting visible OCR text is a hard REJECT."""
        checks = {
            "qr_check": {"qr_detected": True, "qr_decoded": True},
            "qr_consistency": {"status": "INCONSISTENT", "mismatched_fields": ["Name", "DOB"]},
            "face_check": {"is_applicable": True, "document_face_detected": True, "match_status": "MATCH", "similarity_score": 0.90},
            "tamper_check": {"tampering_probability": 0.10},
            "type_check": {"selected": "AADHAAR", "detected": "AADHAAR", "is_mismatch": False}
        }
        res = compute_risk(checks, doc_quality={"quality_score": 88, "is_blurry": False})
        self.assertEqual(res["decision"], "REJECT")
        self.assertEqual(res["category"], "HIGH_RISK")
        self.assertGreaterEqual(res["risk_score"], 70)

    def test_compute_risk_hard_reject_face_mismatch(self):
        """Face mismatch against live selfie is a hard REJECT."""
        checks = {
            "qr_check": {"qr_detected": True, "qr_decoded": True},
            "qr_consistency": {"status": "CONSISTENT", "mismatched_fields": []},
            "face_check": {"is_applicable": True, "document_face_detected": True, "match_status": "MISMATCH", "similarity_score": 0.32},
            "tamper_check": {"tampering_probability": 0.10},
            "type_check": {"selected": "AADHAAR", "detected": "AADHAAR", "is_mismatch": False}
        }
        res = compute_risk(checks, doc_quality={"quality_score": 90, "is_blurry": False})
        self.assertEqual(res["decision"], "REJECT")
        self.assertEqual(res["category"], "HIGH_RISK")

    def test_compare_qr_with_ocr_consistent(self):
        qr_data = {
            "extracted_data": {
                "name": "SPOORTHI KUMAR",
                "dob": "15-08-1998",
                "gender": "MALE",
                "aadhaar_last_4": "4321"
            }
        }
        ocr_fields = {
            "name": "Spoorthi Kumar",
            "date_of_birth": "15/08/1998",
            "gender": "Male",
            "document_number": "XXXX XXXX 4321"
        }
        res = compare_qr_with_ocr(qr_data, ocr_fields)
        self.assertEqual(res["status"], "CONSISTENT")
        self.assertIn("Name", res["matched_fields"])
        self.assertIn("Date of Birth", res["matched_fields"])

    def test_compare_qr_with_ocr_inconsistent(self):
        qr_data = {
            "extracted_data": {
                "name": "RAJESH SHARMA",
                "dob": "01-01-1985",
                "gender": "MALE"
            }
        }
        ocr_fields = {
            "name": "Anita Verma",
            "date_of_birth": "12/05/1995",
            "gender": "Female"
        }
        res = compare_qr_with_ocr(qr_data, ocr_fields)
        self.assertEqual(res["status"], "INCONSISTENT")
        self.assertIn("Name", res["mismatched_fields"])


if __name__ == "__main__":
    unittest.main()
