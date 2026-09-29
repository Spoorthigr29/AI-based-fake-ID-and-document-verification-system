import json
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from documents.models import Document, DocumentQualityAnalysis
from ocr_engine.models import OCRAnalysis
from face_verification.models import FaceVerification
from tamper_detection.models import TamperAnalysis
from identity_verification.models import VerificationResult, IdentityConsistencyCheck
from risk_engine.models import RiskFactor

from risk_engine.services.risk_calculator import DeterministicRiskCalculator
from risk_engine.services.risk_factors import RiskFactorEvaluator
from risk_engine.services.explanation_generator import ExplanationGenerator
from risk_engine.services.ml_risk_predictor import MLRiskPredictor, RiskPredictionResult


class DeterministicRiskCalculatorTests(TestCase):
    """Test suite for deterministic, explainable risk scoring engine."""

    def setUp(self):
        self.calculator = DeterministicRiskCalculator()
        self.evaluator = RiskFactorEvaluator()
        self.client = Client()

        # Low risk synthetic profile
        self.low_risk_signals = {
            "document_quality": 92.0,
            "ocr_confidence": 96.0,
            "face_similarity": 94.0,
            "tampering_probability": 8.0,
            "identity_consistency": 100.0,
            "classification_confidence": 95.0,
            "name_match": 1,
            "dob_match": 1,
            "id_format_valid": 1,
            "address_match": 1,
        }

        # Moderate risk / manual review synthetic profile
        self.manual_review_signals = {
            "document_quality": 62.0,
            "ocr_confidence": 72.0,
            "face_similarity": 68.0,
            "tampering_probability": 32.0,
            "identity_consistency": 75.0,
            "classification_confidence": 65.0,
            "name_match": 1,
            "dob_match": 1,
            "id_format_valid": 1,
            "address_match": 0,
        }

        # High risk synthetic profile
        self.high_risk_signals = {
            "document_quality": 35.0,
            "ocr_confidence": 42.0,
            "face_similarity": 25.0,
            "tampering_probability": 85.0,
            "identity_consistency": 25.0,
            "classification_confidence": 30.0,
            "name_match": 0,
            "dob_match": 0,
            "id_format_valid": 0,
            "address_match": 0,
        }

    def test_low_risk_calculation(self):
        """Verify clean profile calculates to LOW_RISK category (0-30)."""
        result = self.calculator.calculate_risk(self.low_risk_signals)

        self.assertLessEqual(result["risk_score"], 30)
        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertEqual(result["category"], "LOW_RISK")
        self.assertIn("factors", result)
        self.assertGreater(len(result["factors"]), 0)
        self.assertIn("explanation", result)
        self.assertIn("Low overall risk", result["explanation"])

    def test_manual_review_calculation(self):
        """Verify borderline profile calculates to MANUAL_REVIEW category (31-70)."""
        result = self.calculator.calculate_risk(self.manual_review_signals)

        self.assertGreaterEqual(result["risk_score"], 31)
        self.assertLessEqual(result["risk_score"], 70)
        self.assertEqual(result["category"], "MANUAL_REVIEW")
        self.assertIn("Manual", result["explanation"])

    def test_high_risk_calculation(self):
        """Verify fraudulent/anomalous profile calculates to HIGH_RISK category (71-100)."""
        result = self.calculator.calculate_risk(self.high_risk_signals)

        self.assertGreaterEqual(result["risk_score"], 71)
        self.assertLessEqual(result["risk_score"], 100)
        self.assertEqual(result["category"], "HIGH_RISK")
        self.assertIn("Elevated anomaly risk flagged", result["explanation"])

    def test_granular_factor_contributions(self):
        """Verify each factor records name, value, contribution, and severity."""
        factors = self.evaluator.evaluate_all(self.low_risk_signals)
        factor_names = [f.factor_name for f in factors]

        self.assertIn("Face Similarity", factor_names)
        self.assertIn("Tampering Probability", factor_names)
        self.assertIn("Identity Consistency", factor_names)
        self.assertIn("Document Quality", factor_names)
        self.assertIn("OCR Confidence", factor_names)

        # In clean sample, face similarity contribution should be negligible
        face_factor = next(f for f in factors if f.factor_name == "Face Similarity")
        self.assertLessEqual(face_factor.contribution, 2.0)

    def test_evaluate_for_document_database_persistence(self):
        """Verify calculation for a Document model instance creates VerificationResult and RiskFactor rows."""
        # Create test document
        dummy_file = SimpleUploadedFile("id_test.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 200, content_type="image/jpeg")
        doc = Document.objects.create(
            document_type=Document.DOC_TYPE_AADHAAR,
            original_file=dummy_file
        )

        # Attach module analysis results
        DocumentQualityAnalysis.objects.create(
            document=doc,
            classified_type='sample_identity_card',
            classification_confidence=0.95,
            quality_score=94
        )
        OCRAnalysis.objects.create(
            document=doc,
            overall_confidence=0.96,
            raw_text="SAMPLE PERSON"
        )
        FaceVerification.objects.create(
            document=doc,
            document_face_detected=True,
            selfie_face_detected=True,
            similarity_score=0.92,
            confidence=0.92,
            match_status='MATCH'
        )
        TamperAnalysis.objects.create(
            document=doc,
            tampering_probability=0.06,
            risk_level='LOW'
        )
        IdentityConsistencyCheck.objects.create(
            document=doc,
            name_match=True,
            dob_match=True,
            document_number_match=True,
            address_match=True,
            consistency_score=100
        )

        # Run evaluation
        result = self.calculator.evaluate_for_document(doc)

        self.assertEqual(result["category"], "LOW_RISK")
        self.assertLessEqual(result["risk_score"], 30)

        # Verify DB records
        v_res = VerificationResult.objects.get(document=doc)
        self.assertEqual(v_res.overall_risk_score, result["risk_score"])
        self.assertGreater(v_res.risk_factors.count(), 0)

    def test_api_post_risk_score_with_payload(self):
        """Test POST /api/risk-score/<verification_id>/ with JSON payload."""
        url = "/api/risk-score/VX-TEST-000001/"
        response = self.client.post(
            url,
            data=json.dumps(self.low_risk_signals),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("risk_score", data)
        self.assertIn("category", data)
        self.assertIn("factors", data)
        self.assertIn("explanation", data)
        self.assertEqual(data["category"], "LOW_RISK")

    def test_api_post_risk_score_manual_review(self):
        """Test POST /api/risk-score/<verification_id>/ returning MANUAL_REVIEW."""
        url = "/api/risk-score/VX-TEST-000002/"
        response = self.client.post(
            url,
            data=json.dumps(self.manual_review_signals),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "MANUAL_REVIEW")
        self.assertGreaterEqual(data["risk_score"], 31)
        self.assertLessEqual(data["risk_score"], 70)

    def test_api_post_risk_score_high_risk(self):
        """Test POST /api/risk-score/<verification_id>/ returning HIGH_RISK."""
        url = "/api/risk-score/VX-TEST-000003/"
        response = self.client.post(
            url,
            data=json.dumps(self.high_risk_signals),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "HIGH_RISK")
        self.assertGreaterEqual(data["risk_score"], 71)


class MLRiskPredictorTests(TestCase):
    """Test suite for ML Risk Scoring Service and REST API."""

    def setUp(self):
        self.client = Client()
        self.predictor = MLRiskPredictor()

        self.sample_low_risk = {
            "document_quality": 92.0,
            "ocr_confidence": 96.0,
            "face_similarity": 94.0,
            "tampering_probability": 8.0,
            "name_match": 1,
            "dob_match": 1,
            "id_format_valid": 1,
            "address_match": 1,
            "identity_consistency": 100.0
        }

        self.sample_high_risk = {
            "document_quality": 40.0,
            "ocr_confidence": 50.0,
            "face_similarity": 22.0,
            "tampering_probability": 85.0,
            "name_match": 0,
            "dob_match": 0,
            "id_format_valid": 0,
            "address_match": 0,
            "identity_consistency": 25.0
        }

    def test_predictor_low_risk(self):
        """Test clean synthetic input evaluates to LOW_RISK."""
        result = self.predictor.predict_risk(self.sample_low_risk)
        self.assertIsInstance(result, RiskPredictionResult)
        self.assertEqual(result.risk_category, "LOW_RISK")
        self.assertGreater(result.confidence, 0.50)

    def test_predictor_high_risk(self):
        """Test anomalous input receives MANUAL_REVIEW or HIGH_RISK."""
        result = self.predictor.predict_risk(self.sample_high_risk)
        self.assertIn(result.risk_category, ["MANUAL_REVIEW", "HIGH_RISK"])
        self.assertGreater(result.confidence, 0.50)

    def test_predictor_missing_feature_raises_error(self):
        """Test missing required feature raises informative ValueError."""
        incomplete_sample = self.sample_low_risk.copy()
        del incomplete_sample["face_similarity"]

        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_risk(incomplete_sample)
        self.assertIn("Missing required feature", str(ctx.exception))

    def test_api_post_ml_risk_score_success(self):
        """Test POST /api/risk-score/ml/ returns expected response format."""
        url = "/api/risk-score/ml/"
        response = self.client.post(
            url,
            data=json.dumps(self.sample_low_risk),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data.get("success"))
        self.assertIn("risk_category", data)
        self.assertIn("confidence", data)
        self.assertIn("model", data)
        self.assertEqual(data["risk_category"], "LOW_RISK")
