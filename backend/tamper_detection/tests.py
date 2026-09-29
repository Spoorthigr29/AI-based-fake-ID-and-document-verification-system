import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from documents.models import Document
from tamper_detection.models import TamperAnalysis
from tamper_detection.services.ela_analyzer import ELAAnalyzer, ELAResult
from tamper_detection.services.artifact_analyzer import ArtifactAnalyzer, ArtifactResult
from tamper_detection.services.region_detector import RegionAnomalyDetector, RegionDetectionResult
from tamper_detection.services.tamper_analyzer import TamperAnalyzerService, HeuristicTamperModel

def generate_clean_synthetic_doc(size=(600, 400)):
    """Generate clean synthetic identity document image with uniform noise and gradient."""
    img = Image.new('RGB', size, color=(248, 249, 250))
    draw = ImageDraw.Draw(img)

    # Header Banner
    draw.rectangle((20, 20, 580, 70), fill=(30, 41, 59))
    draw.text((35, 35), "NATIONAL SAMPLE IDENTITY CARD", fill=(255, 255, 255))

    # Text fields
    draw.text((40, 100), "NAME: SAMPLE CITIZEN", fill=(15, 23, 42))
    draw.text((40, 135), "DOB: 1995-08-15", fill=(15, 23, 42))
    draw.text((40, 170), "DOC ID: 8940 2190 4421", fill=(15, 23, 42))
    draw.text((40, 205), "ADDRESS: 123 SAMPLE ROAD", fill=(15, 23, 42))

    # Photo Box
    draw.rectangle((420, 100, 560, 260), fill=(220, 225, 230), outline=(100, 110, 120), width=1)
    draw.ellipse((450, 125, 530, 205), fill=(235, 195, 170)) # Portrait silhouette

    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG', quality=92)
    byte_arr.seek(0)
    return byte_arr.getvalue()

def generate_tampered_synthetic_doc(size=(600, 400)):
    """
    Generate tampered synthetic document containing spliced photo patch and copy-pasted text anomaly.
    """
    clean_bytes = generate_clean_synthetic_doc(size=size)
    img = Image.open(io.BytesIO(clean_bytes))
    draw = ImageDraw.Draw(img)

    # 1. Spliced photo region with unnatural sharp step gradient
    draw.rectangle((415, 95, 565, 265), fill=(255, 100, 100), outline=(0, 0, 0), width=4)
    draw.text((430, 170), "SPLICED", fill=(255, 255, 255))

    # 2. Cloned / Pasted text box with mismatched background
    draw.rectangle((35, 165, 300, 195), fill=(255, 255, 255), outline=(255, 0, 0), width=3)
    draw.text((40, 170), "ALTERED ID: 9999 9999 9999", fill=(0, 0, 0))

    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG', quality=65) # Lower quality re-compression introduces high ELA variance
    byte_arr.seek(0)
    return byte_arr.getvalue()

class TamperDetectionTests(TestCase):
    """Test suite for Error Level Analysis, Artifacts, Boundary Anomalies, and Tampering API."""

    def setUp(self):
        self.client = Client()
        self.ela_analyzer = ELAAnalyzer()
        self.artifact_analyzer = ArtifactAnalyzer()
        self.region_detector = RegionAnomalyDetector()

        clean_bytes = generate_clean_synthetic_doc()
        self.clean_doc = Document.objects.create(
            verification_id="VX-2026-CLEAN01",
            original_file=SimpleUploadedFile("clean_doc.jpg", clean_bytes, content_type="image/jpeg"),
            file_hash="clean_hash_123",
            processing_status=Document.STATUS_UPLOADED
        )

        tampered_bytes = generate_tampered_synthetic_doc()
        self.tampered_doc = Document.objects.create(
            verification_id="VX-2026-TAMPER01",
            original_file=SimpleUploadedFile("tampered_doc.jpg", tampered_bytes, content_type="image/jpeg"),
            file_hash="tamper_hash_123",
            processing_status=Document.STATUS_UPLOADED
        )

    def test_ela_analyzer_execution(self):
        """Test Error Level Analysis returns valid metrics and difference array."""
        clean_bytes = generate_clean_synthetic_doc()
        result: ELAResult = self.ela_analyzer.analyze(clean_bytes)

        self.assertIsInstance(result.ela_score, float)
        self.assertGreaterEqual(result.ela_score, 0.0)
        self.assertLessEqual(result.ela_score, 1.0)
        self.assertIsNotNone(result.ela_image_array)

    def test_clean_document_tamper_screening(self):
        """Test that a clean synthetic document receives low tampering probability."""
        service = TamperAnalyzerService()
        result = service.process_document(self.clean_doc)

        self.assertIn("tampering_probability", result)
        self.assertIn("risk_level", result)
        self.assertIn("signals", result)
        self.assertIn("suspicious_regions", result)

        self.assertIn(result["risk_level"], [TamperAnalysis.RISK_LOW, TamperAnalysis.RISK_MODERATE])
        self.assertLess(result["tampering_probability"], 0.65)

    def test_tampered_document_detection(self):
        """Test that an altered document triggers tampering signals and is flagged for review."""
        service = TamperAnalyzerService()
        result = service.process_document(self.tampered_doc)

        self.assertGreater(result["tampering_probability"], 0.40)
        self.assertIn(result["risk_level"], [TamperAnalysis.RISK_MODERATE, TamperAnalysis.RISK_HIGH])
        self.assertGreater(len(result["signals"]), 0)

        # Confirm database persistence
        record = TamperAnalysis.objects.get(document=self.tampered_doc)
        self.assertIsNotNone(record)
        self.assertIsNotNone(record.annotated_image)

    def test_tamper_api_post_endpoint(self):
        """Test POST /api/tamper/analyze/<verification_id>/ returns expected schema."""
        url = f"/api/tamper/analyze/{self.tampered_doc.verification_id}/"
        response = self.client.post(url, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("tampering_probability", data)
        self.assertIn("risk_level", data)
        self.assertIn("signals", data)
        self.assertIn("suspicious_regions", data)
        self.assertIsInstance(data["tampering_probability"], float)
        self.assertIsInstance(data["signals"], list)

    def test_tamper_api_get_endpoint(self):
        """Test GET /api/tamper/analyze/<verification_id>/ returns analysis record."""
        # Run analysis first
        self.client.post(f"/api/tamper/analyze/{self.clean_doc.verification_id}/")

        response = self.client.get(f"/api/tamper/analyze/{self.clean_doc.verification_id}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("risk_level", data)

    def test_tamper_results_html_view(self):
        """Test HTML view /tamper/results/<verification_id>/ renders without error."""
        response = self.client.get(reverse('tamper_detection:results', kwargs={'verification_id': self.clean_doc.verification_id}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "DOCUMENT TAMPERING ANALYSIS")
