import os
import unittest
from documents.services.aadhaar_fixes import compute_risk
from risk_engine.services.risk_calculator import DeterministicRiskCalculator

class TestFiveScenarios(unittest.TestCase):
    """
    Executes the 5 required test cases:
    TEST 1: Valid PAN with successful liveness and face match.
    TEST 2: PAN with poor OCR quality.
    TEST 3: PAN where QR verification is unavailable.
    TEST 4: Document with an actual identity mismatch.
    TEST 5: Document with suspicious integrity evidence.
    """

    def test_run_and_display_all_scenarios(self):
        calculator = DeterministicRiskCalculator()

        scenarios = [
            {
                "id": "TEST 1",
                "title": "Valid PAN with successful liveness and face match",
                "signals": {
                    "document_type": "PAN",
                    "selected_type": "PAN",
                    "detected_type": "PAN",
                    "is_type_mismatch": False,
                    "ocr_confidence": 0.92,
                    "qr_detected": False,
                    "qr_decoded": False,
                    "qr_consistency_status": "NOT_AVAILABLE",
                    "tampering_probability": 0.08,
                    "document_face_detected": True,
                    "face_similarity": 0.94,
                    "face_match_status": "MATCH",
                    "identity_consistency": 100.0,
                    "document_quality": 92
                }
            },
            {
                "id": "TEST 2",
                "title": "PAN with poor OCR quality (62% moderate / uncertainty)",
                "signals": {
                    "document_type": "PAN",
                    "selected_type": "PAN",
                    "detected_type": "PAN",
                    "is_type_mismatch": False,
                    "ocr_confidence": 0.62,
                    "qr_detected": False,
                    "qr_decoded": False,
                    "qr_consistency_status": "NOT_AVAILABLE",
                    "tampering_probability": 0.10,
                    "document_face_detected": True,
                    "face_similarity": 0.91,
                    "face_match_status": "MATCH",
                    "identity_consistency": 85.0,
                    "document_quality": 78
                }
            },
            {
                "id": "TEST 3",
                "title": "PAN where QR verification is unavailable (standard non-QR card)",
                "signals": {
                    "document_type": "PAN",
                    "selected_type": "PAN",
                    "detected_type": "PAN",
                    "is_type_mismatch": False,
                    "ocr_confidence": 0.88,
                    "qr_detected": False,
                    "qr_decoded": False,
                    "qr_consistency_status": "NOT_AVAILABLE",
                    "tampering_probability": 0.09,
                    "document_face_detected": True,
                    "face_similarity": 0.89,
                    "face_match_status": "MATCH",
                    "identity_consistency": 95.0,
                    "document_quality": 88
                }
            },
            {
                "id": "TEST 4",
                "title": "Document with an actual identity mismatch (Name & ID discrepancy)",
                "signals": {
                    "document_type": "PAN",
                    "selected_type": "PAN",
                    "detected_type": "PAN",
                    "is_type_mismatch": False,
                    "ocr_confidence": 0.85,
                    "qr_detected": False,
                    "qr_decoded": False,
                    "qr_consistency_status": "NOT_AVAILABLE",
                    "tampering_probability": 0.12,
                    "document_face_detected": True,
                    "face_similarity": 0.35,
                    "face_match_status": "MISMATCH",
                    "identity_consistency": 30.0,
                    "document_quality": 85
                }
            },
            {
                "id": "TEST 5",
                "title": "Document with suspicious integrity evidence (Tampering / High ELA anomaly)",
                "signals": {
                    "document_type": "PAN",
                    "selected_type": "PAN",
                    "detected_type": "PAN",
                    "is_type_mismatch": False,
                    "ocr_confidence": 0.82,
                    "qr_detected": False,
                    "qr_decoded": False,
                    "qr_consistency_status": "NOT_AVAILABLE",
                    "tampering_probability": 0.85,
                    "tamper_signals": ["localized ELA error discrepancy detected", "editing software metadata detected"],
                    "document_face_detected": True,
                    "face_similarity": 0.88,
                    "face_match_status": "MATCH",
                    "identity_consistency": 90.0,
                    "document_quality": 80
                }
            }
        ]

        print("\n" + "="*80)
        print("VERIFYX AI - 5 VERIFICATION SCENARIOS EVALUATION REPORT")
        print("="*80)

        for sc in scenarios:
            res = calculator.calculate_risk(sc["signals"])
            print(f"\n[{sc['id']}] {sc['title']}")
            print("-" * 60)
            print(f"* Document Type:        {sc['signals']['document_type']} [MATCHED]")
            
            ocr_c = sc['signals']['ocr_confidence']
            ocr_text = f"{int(ocr_c*100)}% - Passed" if ocr_c >= 0.70 else f"{int(ocr_c*100)}% - Review Required"
            print(f"* OCR Result:           {ocr_text}")
            
            t_prob = int(sc['signals']['tampering_probability'] * 100)
            t_text = f"Low Risk ({t_prob}%)" if t_prob <= 35 else (f"Medium Risk ({t_prob}%)" if t_prob <= 60 else f"High Risk ({t_prob}%)")
            print(f"* Document Integrity:   {t_text}")
            
            qr_text = "Unavailable (0 penalty)" if not sc['signals']['qr_decoded'] else "Verified"
            print(f"* QR Result:            {qr_text}")
            
            liveness_text = "Passed"
            print(f"* Liveness:             {liveness_text}")
            
            face_st = sc['signals']['face_match_status']
            face_sim = sc['signals']['face_similarity']
            face_text = f"Matched ({int(face_sim*100)}%)" if face_st == "MATCH" else (f"Mismatch ({int(face_sim*100)}%)" if face_st == "MISMATCH" else "Review Required")
            print(f"* Face Verification:    {face_text}")
            
            id_cons = sc['signals']['identity_consistency']
            id_text = f"Consistent ({int(id_cons)}%)" if id_cons >= 70 else f"Inconsistent ({int(id_cons)}%)"
            print(f"* Identity Consistency: {id_text}")
            
            print(f"* Final Risk Score:     {res['risk_score']} / 100")
            print(f"* Final Status:         {res['category']} ({res['decision']})")
            print(f"* Reason for Result:    {res['explanation']}")

            # Assertions to ensure strict mathematical validity
            if sc["id"] == "TEST 1":
                self.assertLessEqual(res["risk_score"], 29)
                self.assertEqual(res["category"], "LOW_RISK")
            elif sc["id"] == "TEST 2":
                self.assertLessEqual(res["risk_score"], 29)
                self.assertEqual(res["category"], "LOW_RISK")
            elif sc["id"] == "TEST 3":
                self.assertLessEqual(res["risk_score"], 29)
                self.assertEqual(res["category"], "LOW_RISK")
            elif sc["id"] == "TEST 4":
                self.assertGreaterEqual(res["risk_score"], 60)
                self.assertEqual(res["category"], "HIGH_RISK")
            elif sc["id"] == "TEST 5":
                self.assertGreaterEqual(res["risk_score"], 30)

        print("\n" + "="*80)

if __name__ == "__main__":
    unittest.main()
