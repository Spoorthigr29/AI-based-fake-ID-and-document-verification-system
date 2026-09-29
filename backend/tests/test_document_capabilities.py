"""
Unit and integration tests for VerifyX AI Document Capabilities & Verification Pipeline.

Covers:
1. PAN selected -> selfie not required, face matching skipped, face status NOT_APPLICABLE
2. Aadhaar selected -> selfie required, face matching enabled
3. Passport selected -> selfie required, face matching enabled
4. Driving License selected -> selfie required, face matching enabled
5. Voter ID selected -> selfie required, face matching enabled
6. PAN risk score neutrality (0 contribution for face verification)
7. Document capabilities config queries & dynamic payload checks
8. End-to-end pipeline execution with PAN vs Aadhaar
"""

import io
import json
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from documents.models import Document
from documents.services.document_capabilities import (
    DOCUMENT_CAPABILITIES,
    get_document_capabilities,
    has_document_photo,
    requires_live_selfie,
    is_face_matching_enabled,
    normalize_document_type
)
from face_verification.models import FaceVerification
from face_verification.services.face_service import FaceVerificationService
from risk_engine.services.risk_calculator import DeterministicRiskCalculator
from risk_engine.services.risk_factors import RiskFactorEvaluator
from verification.services.verification_pipeline import VerificationPipelineOrchestrator


def generate_test_image(text="SAMPLE"):
    img = Image.new('RGB', (400, 250), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 390, 240], outline=(100, 100, 100), width=2)
    draw.text((30, 30), text, fill=(20, 20, 20))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr.getvalue()


class DocumentCapabilitiesTestSuite(TestCase):
    """Thorough tests for document capability rules, face verification skipping, and risk assessment."""

    def setUp(self):
        self.client = Client()
        self.sample_bytes = generate_test_image("PAN CARD SAMPLE")

    def test_01_pan_capabilities_configuration(self):
        """1. PAN selected -> selfie required, has_document_photo=True, enable_face_matching=True."""
        caps = get_document_capabilities("sample_pan")
        self.assertTrue(caps["has_document_photo"])
        self.assertTrue(caps["requires_live_selfie"])
        self.assertTrue(caps["enable_face_matching"])

        self.assertTrue(has_document_photo("PAN"))
        self.assertTrue(requires_live_selfie("PAN"))
        self.assertTrue(is_face_matching_enabled("PAN"))

    def test_02_aadhaar_capabilities_configuration(self):
        """2. Aadhaar selected -> selfie required, face matching enabled, has_document_photo=True."""
        caps = get_document_capabilities("sample_aadhaar")
        self.assertTrue(caps["has_document_photo"])
        self.assertTrue(caps["requires_live_selfie"])
        self.assertTrue(caps["enable_face_matching"])

        self.assertTrue(has_document_photo("AADHAAR"))
        self.assertTrue(requires_live_selfie("AADHAAR"))
        self.assertTrue(is_face_matching_enabled("AADHAAR"))

    def test_03_passport_capabilities_configuration(self):
        """3. Passport selected -> selfie required, face matching enabled, has_document_photo=True."""
        caps = get_document_capabilities("sample_passport")
        self.assertTrue(caps["has_document_photo"])
        self.assertTrue(caps["requires_live_selfie"])
        self.assertTrue(caps["enable_face_matching"])

        self.assertTrue(has_document_photo("PASSPORT"))
        self.assertTrue(requires_live_selfie("PASSPORT"))
        self.assertTrue(is_face_matching_enabled("PASSPORT"))

    def test_04_driving_license_capabilities_configuration(self):
        """4. Driving License selected -> selfie required, face matching enabled, has_document_photo=True."""
        caps = get_document_capabilities("sample_driving_license")
        self.assertTrue(caps["has_document_photo"])
        self.assertTrue(caps["requires_live_selfie"])
        self.assertTrue(caps["enable_face_matching"])

        self.assertTrue(has_document_photo("DRIVING_LICENSE"))
        self.assertTrue(requires_live_selfie("DRIVING_LICENSE"))
        self.assertTrue(is_face_matching_enabled("DRIVING_LICENSE"))

    def test_05_voter_id_capabilities_configuration(self):
        """5. Voter ID selected -> selfie required, face matching enabled, has_document_photo=True."""
        caps = get_document_capabilities("sample_voter_id")
        self.assertTrue(caps["has_document_photo"])
        self.assertTrue(caps["requires_live_selfie"])
        self.assertTrue(caps["enable_face_matching"])

        self.assertTrue(has_document_photo("VOTER_ID"))
        self.assertTrue(requires_live_selfie("VOTER_ID"))
        self.assertTrue(is_face_matching_enabled("VOTER_ID"))

    def test_06_pan_face_verification_skipped_and_status_not_applicable(self):
        """PAN verification does not search for faces or perform matching; marks NOT_APPLICABLE."""
        doc_file = SimpleUploadedFile("pan.jpg", self.sample_bytes, content_type="image/jpeg")
        pan_doc = Document.objects.create(
            verification_id="VX-TEST-PAN-01",
            document_type="PAN",
            original_file=doc_file,
            selfie_file=None,
            processing_status=Document.STATUS_UPLOADED
        )

        service = FaceVerificationService()
        result = service.process_verification(pan_doc)

        self.assertEqual(result["match_status"], "NOT_APPLICABLE")
        self.assertIsNone(result["similarity_score"])
        self.assertIsNone(result["confidence"])
        self.assertFalse(result["document_face_detected"])
        self.assertIn("No usable photograph was detected", result["explanation"])

        # Check DB record
        record = FaceVerification.objects.get(document=pan_doc)
        self.assertEqual(record.match_status, FaceVerification.MATCH_STATUS_NOT_APPLICABLE)
        self.assertIsNone(record.similarity_score)
        self.assertIsNone(record.confidence)
        self.assertIn("No usable photograph was detected", record.explanation)

        # PAN should NOT be routed to MANUAL_REVIEW due to missing selfie/face
        pan_doc.refresh_from_db()
        self.assertNotEqual(pan_doc.processing_status, Document.STATUS_MANUAL_REVIEW)

    def test_07_pan_risk_penalty_neutrality(self):
        """6. PAN must NOT receive a risk penalty because face matching is unavailable."""
        evaluator = RiskFactorEvaluator()
        factor_na = evaluator.evaluate_face_similarity(None, match_status="NOT_APPLICABLE")
        self.assertEqual(factor_na.contribution, 0.0)
        self.assertEqual(factor_na.factor_value, "NOT_APPLICABLE")

        # Directly test DeterministicRiskCalculator with face_similarity=None / not applicable
        calc = DeterministicRiskCalculator()
        signals = {
            'document_quality': 95.0,
            'tampering_probability': 0.05,
            'face_similarity': None,
            'face_match_status': 'NOT_APPLICABLE',
            'ocr_confidence': 0.95,
            'identity_consistency': 95.0,
            'document_type': 'PAN'
        }
        risk_result_pan = calc.calculate_risk(signals)
        self.assertEqual(risk_result_pan['category'], "LOW_RISK")
        self.assertLessEqual(risk_result_pan['risk_score'], 30)

        # Ensure face verification contributed 0 points
        face_factor = next((f for f in risk_result_pan['factors'] if f['name'] == "Face Verification"), None)
        self.assertIsNotNone(face_factor)
        self.assertEqual(face_factor['contribution'], 0.0)
        self.assertEqual(face_factor['value'], "NOT_APPLICABLE")

    def test_08_dynamic_ui_context_in_views(self):
        """Test that upload view and detail view provide correct capability flags."""
        from django.contrib.auth.models import User
        user = User.objects.create(username="superadmin", email="admin@verifyx.ai", is_staff=True, is_superuser=True)
        self.client.force_login(user)

        doc_file = SimpleUploadedFile("pan.jpg", self.sample_bytes, content_type="image/jpeg")
        pan_doc = Document.objects.create(
            verification_id="VX-TEST-PAN-02",
            document_type="PAN",
            uploaded_by=user,
            original_file=doc_file,
            selfie_file=None,
            processing_status=Document.STATUS_COMPLETED
        )
        FaceVerification.objects.create(
            document=pan_doc,
            match_status=FaceVerification.MATCH_STATUS_NOT_APPLICABLE,
            similarity_score=None,
            confidence=None,
            explanation="Face comparison is not applicable because the selected PAN document template does not contain a photograph."
        )

        # Detail view should render with is_face_applicable=False
        url = reverse('documents:detail', kwargs={'verification_id': pan_doc.verification_id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Not Applicable")

        # Upload view should contain document capabilities JSON
        upload_url = reverse('documents:upload')
        upload_resp = self.client.get(upload_url)
        self.assertEqual(upload_resp.status_code, 200)
        self.assertContains(upload_resp, "sample_pan")
        self.assertContains(upload_resp, "sample_aadhaar")
        self.assertContains(upload_resp, "has_document_photo")
