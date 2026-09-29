"""
VerifyX AI - WEBP Image Upload & Multi-Format Validation Test Suite
===================================================================
Validates:
1. Upload and processing of WEBP documents
2. Upload and processing of JPG & PNG documents
3. Upload of WEBP selfies
4. Rejection of unsupported extensions / formats
5. Rejection of corrupted image data
6. Rejection of oversized files (> 10MB)
7. End-to-end multi-modal pipeline execution on WEBP documents
"""

import io
import os
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User

from documents.models import Document
from documents.validators import (
    validate_uploaded_document,
    validate_uploaded_selfie,
    MAX_FILE_SIZE_BYTES
)
from documents.services.quality_analyzer import DocumentQualityAnalyzer
from documents.services.document_classifier import DocumentClassifier
from ocr_engine.services.ocr_service import OCREngineService
from tamper_detection.services.tamper_analyzer import TamperAnalyzerService
from face_verification.services.face_service import FaceVerificationService
from verification.services.verification_pipeline import VerificationPipelineOrchestrator


def generate_sample_image(format='WEBP', text="SAMPLE IDENTITY", size=(500, 320), color=(240, 242, 245)):
    """Generate an in-memory valid raster image in specified format."""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([15, 15, size[0]-15, size[1]-15], outline=(60, 80, 120), width=3)
    draw.text((30, 40), f"VERIFYX ID - {text}", fill=(20, 30, 50))
    draw.text((30, 90), "Name: SAMPLE CITIZEN", fill=(40, 40, 40))
    draw.text((30, 130), "DOB: 15/08/1990", fill=(40, 40, 40))
    draw.text((30, 170), "ID: ABCDE1234F", fill=(40, 40, 40))

    # Add face circle geometry
    draw.ellipse([size[0]-140, 40, size[0]-30, 180], fill=(230, 200, 180), outline=(50, 40, 30), width=2)

    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


class WebpUploadValidationTests(TestCase):
    """Thorough tests for WEBP and multi-format upload validation and processing."""

    def setUp(self):
        self.client = Client()
        self.admin_user = User.objects.create_superuser(
            username="webp_admin",
            password="password123",
            email="webp_admin@verifyx.ai"
        )
        self.client.force_login(self.admin_user)

        self.webp_bytes = generate_sample_image(format='WEBP', text="WEBP TEST")
        self.jpg_bytes = generate_sample_image(format='JPEG', text="JPEG TEST")
        self.png_bytes = generate_sample_image(format='PNG', text="PNG TEST")

    def test_01_validate_uploaded_document_webp_success(self):
        """WEBP document bytes pass validator cleanly."""
        uploaded = SimpleUploadedFile("id_document.webp", self.webp_bytes, content_type="image/webp")
        self.assertTrue(validate_uploaded_document(uploaded))

    def test_02_validate_uploaded_selfie_webp_success(self):
        """WEBP selfie bytes pass validator cleanly."""
        uploaded = SimpleUploadedFile("applicant_selfie.webp", self.webp_bytes, content_type="image/webp")
        self.assertTrue(validate_uploaded_selfie(uploaded))

    def test_03_validate_uploaded_document_jpg_and_png_success(self):
        """JPG and PNG formats continue to pass validation."""
        jpg_upload = SimpleUploadedFile("id_document.jpg", self.jpg_bytes, content_type="image/jpeg")
        png_upload = SimpleUploadedFile("id_document.png", self.png_bytes, content_type="image/png")
        self.assertTrue(validate_uploaded_document(jpg_upload))
        self.assertTrue(validate_uploaded_document(png_upload))

    def test_04_reject_unsupported_format(self):
        """Unsupported file extension raises ValidationError with informative message."""
        fake_txt = SimpleUploadedFile("document.txt", b"Plain text file content", content_type="text/plain")
        with self.assertRaises(ValidationError) as ctx:
            validate_uploaded_document(fake_txt)
        self.assertIn("Unsupported file format", str(ctx.exception))

    def test_05_reject_corrupted_image(self):
        """Corrupted WEBP image data with correct header is caught and rejected."""
        corrupted_bytes = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 50
        corrupted_file = SimpleUploadedFile("broken.webp", corrupted_bytes, content_type="image/webp")
        with self.assertRaises(ValidationError) as ctx:
            validate_uploaded_document(corrupted_file)
        self.assertIn("Corrupted image", str(ctx.exception))

    def test_06_reject_oversized_file(self):
        """File larger than 10MB is rejected."""
        large_bytes = b"\x00" * (MAX_FILE_SIZE_BYTES + 1024)
        large_file = SimpleUploadedFile("huge.webp", large_bytes, content_type="image/webp")
        with self.assertRaises(ValidationError) as ctx:
            validate_uploaded_document(large_file)
        self.assertIn("File size exceeds", str(ctx.exception))

    def test_07_quality_analyzer_with_webp(self):
        """DocumentQualityAnalyzer successfully processes WEBP file bytes and array."""
        res = DocumentQualityAnalyzer.analyze_quality(self.webp_bytes)
        self.assertIn("quality_score", res)
        self.assertGreater(res["quality_score"], 0)
        self.assertIn("metrics", res)

    def test_08_tamper_analyzer_with_webp(self):
        """Tamper detection engines (ELA and Heuristic Model) handle WEBP images without errors."""
        from tamper_detection.services.ela_analyzer import ELAAnalyzer
        from tamper_detection.services.tamper_analyzer import HeuristicTamperModel

        ela = ELAAnalyzer()
        ela_res = ela.analyze(self.webp_bytes)
        self.assertGreaterEqual(ela_res.ela_score, 0.0)
        self.assertIsNotNone(ela_res.signals)

        model = HeuristicTamperModel()
        pred = model.predict(self.webp_bytes)
        self.assertIn("tampering_probability", pred)
        self.assertIn("risk_level", pred)

    def test_09_end_to_end_webp_document_upload_view(self):
        """Full HTTP upload with WEBP document and selfie initiates verification."""
        doc_upload = SimpleUploadedFile("specimen.webp", self.webp_bytes, content_type="image/webp")
        selfie_upload = SimpleUploadedFile("selfie.webp", self.webp_bytes, content_type="image/webp")

        url = reverse('documents:upload')
        response = self.client.post(url, {
            'document_type': 'PAN',
            'original_file': doc_upload,
            'applicant_selfie': selfie_upload,
        })

        self.assertEqual(response.status_code, 302)
        redirect_url = response.url
        self.assertIn('/documents/processing/', redirect_url)

        # Confirm Document instance was created with original .webp file
        created_doc = Document.objects.latest('id')
        self.assertTrue(created_doc.original_file.name.endswith('.webp'))
        self.assertTrue(os.path.exists(created_doc.original_file.path))
