import io
import json
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from documents.models import Document, DocumentQualityAnalysis
from ocr_engine.models import OCRAnalysis, ExtractedField
from face_verification.models import FaceVerification
from tamper_detection.models import TamperAnalysis
from identity_verification.models import VerificationResult, IdentityConsistencyCheck
from risk_engine.models import RiskFactor
from verification.services.verification_pipeline import VerificationPipelineOrchestrator


def create_sample_id_image(text="GOVERNMENT OF INDIA\nNAME: JOHN DOE\nDOB: 1995-08-15\nID: 9912 3456 7890"):
    """Helper to generate an in-memory sample document JPEG."""
    image = Image.new('RGB', (800, 500), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 780, 480], outline=(100, 100, 100), width=3)
    draw.rectangle([40, 60, 180, 220], fill=(200, 200, 200), outline=(50, 50, 50)) # simulated face photo box
    draw.text((220, 80), text, fill=(0, 0, 0))
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return SimpleUploadedFile("sample_id.jpg", byte_arr.getvalue(), content_type="image/jpeg")

def create_sample_selfie_image():
    """Helper to generate an in-memory selfie JPEG."""
    image = Image.new('RGB', (400, 400), color=(220, 230, 240))
    draw = ImageDraw.Draw(image)
    draw.ellipse([100, 80, 300, 320], fill=(240, 200, 180), outline=(50, 50, 50)) # simulated face
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return SimpleUploadedFile("selfie.jpg", byte_arr.getvalue(), content_type="image/jpeg")


class VerificationPipelineTests(TestCase):
    """End-to-End test suite for unified verification pipeline."""

    def setUp(self):
        self.client = Client()
        self.orchestrator = VerificationPipelineOrchestrator()

        self.doc_file = create_sample_id_image()
        self.selfie_file = create_sample_selfie_image()

        self.document = Document.objects.create(
            document_type=Document.DOC_TYPE_AADHAAR,
            original_file=self.doc_file,
            selfie_file=self.selfie_file,
            processing_status=Document.STATUS_UPLOADED
        )

    def test_end_to_end_pipeline_orchestrator_execution(self):
        """Verify full 12-stage pipeline runs sequentially and persists all models."""
        result = self.orchestrator.run_pipeline(self.document)

        self.assertIn("verification_id", result)
        self.assertEqual(result["verification_id"], self.document.verification_id)
        self.assertIn("status", result)
        self.assertIn("risk_score", result)
        self.assertIn("category", result)
        self.assertIn("stages", result)
        self.assertGreaterEqual(len(result["stages"]), 12)

        # Verify all stages recorded timestamps and completion
        for stage in result["stages"]:
            self.assertIn("stage", stage)
            self.assertIn("name", stage)
            self.assertIn("status", stage)
            self.assertIn("timestamp", stage)

        # Verify DB Records Created
        self.assertTrue(DocumentQualityAnalysis.objects.filter(document=self.document).exists())
        self.assertTrue(OCRAnalysis.objects.filter(document=self.document).exists())
        self.assertTrue(FaceVerification.objects.filter(document=self.document).exists())
        self.assertTrue(TamperAnalysis.objects.filter(document=self.document).exists())
        self.assertTrue(IdentityConsistencyCheck.objects.filter(document=self.document).exists())
        self.assertTrue(VerificationResult.objects.filter(document=self.document).exists())
        self.assertGreater(RiskFactor.objects.filter(verification_result__document=self.document).count(), 0)

    def test_graceful_handling_without_selfie(self):
        """Pipeline must not crash when no selfie is attached."""
        single_doc = Document.objects.create(
            document_type=Document.DOC_TYPE_PAN,
            original_file=create_sample_id_image(),
            selfie_file=None,
            processing_status=Document.STATUS_UPLOADED
        )

        result = self.orchestrator.run_pipeline(single_doc)
        self.assertIn("status", result)
        self.assertIn("risk_score", result)
        self.assertEqual(result["verification_id"], single_doc.verification_id)

    def test_api_start_verification_endpoint(self):
        """Test POST /api/verification/start/<verification_id>/ returns expected payload."""
        url = f"/api/verification/start/{self.document.verification_id}/"
        response = self.client.post(
            url,
            data=json.dumps({"name": "JOHN DOE", "dob": "1995-08-15"}),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check exact required response keys
        self.assertEqual(data["verification_id"], self.document.verification_id)
        self.assertIn("status", data)
        self.assertIn("risk_score", data)
        self.assertIn("category", data)
        self.assertIsInstance(data["risk_score"], int)
        self.assertIn(data["category"], ["LOW_RISK", "MANUAL_REVIEW", "HIGH_RISK"])

    def test_api_start_verification_not_found(self):
        """Test POST /api/verification/start/<non_existent_id>/ returns 404."""
        url = "/api/verification/start/VX-NONEXISTENT-999999/"
        response = self.client.post(url)
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("error", data)
