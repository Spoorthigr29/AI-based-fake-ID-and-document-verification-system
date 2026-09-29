import io
import cv2
import numpy as np
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings

from documents.models import Document
from face_verification.models import FaceVerification
from face_verification.services.face_detector import FaceDetector, FaceDetectionResult
from face_verification.services.face_embedder import StructuralBiometricEmbedder, FaceEmbedder
from face_verification.services.face_service import FaceVerificationService

def generate_synthetic_face_image(size=(400, 400), background_color=(230, 230, 235), face_type='person_a'):
    """
    Generate a clean in-memory synthetic portrait containing facial geometry
    recognizable by edge detectors and OpenCV cascades.
    """
    img = Image.new('RGB', size, color=background_color)
    draw = ImageDraw.Draw(img)

    # Face Oval
    face_box = (100, 60, 300, 320)
    skin_tone = (245, 215, 190) if face_type == 'person_a' else (220, 185, 160)
    draw.ellipse(face_box, fill=skin_tone, outline=(70, 50, 40), width=2)

    # Hair
    hair_color = (40, 25, 15) if face_type == 'person_a' else (75, 45, 20)
    draw.chord((90, 40, 310, 180), start=180, end=360, fill=hair_color)

    # Eyes
    # Left Eye
    draw.ellipse((140, 140, 175, 165), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.ellipse((152, 145, 165, 158), fill=(30, 20, 10))
    draw.line((135, 130, 180, 133), fill=(40, 20, 10), width=3) # Eyebrow

    # Right Eye
    draw.ellipse((225, 140, 260, 165), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.ellipse((235, 145, 248, 158), fill=(30, 20, 10))
    draw.line((220, 133, 265, 130), fill=(40, 20, 10), width=3) # Eyebrow

    # Nose
    draw.line((200, 160, 195, 210), fill=(160, 110, 90), width=2)
    draw.line((195, 210, 208, 210), fill=(160, 110, 90), width=2)

    # Mouth
    draw.arc((160, 230, 240, 270), start=20, end=160, fill=(180, 50, 50), width=3)

    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG', quality=95)
    byte_arr.seek(0)
    return byte_arr.getvalue()

def generate_blank_image(size=(300, 300)):
    """Generate blank image with no facial features."""
    img = Image.new('RGB', size, color=(255, 255, 255))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr.getvalue()

class FaceVerificationTests(TestCase):
    """Test suite for Face Detection, Feature Embedding, Verification Pipeline, and API."""

    def setUp(self):
        self.client = Client()
        self.detector = FaceDetector()
        self.embedder = StructuralBiometricEmbedder()

        # Create sample document and selfie uploads
        doc_bytes = generate_synthetic_face_image(face_type='person_a')
        selfie_bytes = generate_synthetic_face_image(face_type='person_a')

        self.doc_file = SimpleUploadedFile("id_doc.jpg", doc_bytes, content_type="image/jpeg")
        self.selfie_file = SimpleUploadedFile("selfie.jpg", selfie_bytes, content_type="image/jpeg")

        self.document = Document.objects.create(
            verification_id="VX-2026-TEST01",
            document_type=Document.DOC_TYPE_AADHAAR,
            original_file=self.doc_file,
            selfie_file=self.selfie_file,
            file_hash="test_doc_hash_123",
            selfie_hash="test_selfie_hash_123",
            processing_status=Document.STATUS_UPLOADED
        )

    def test_structural_embedder_similarity(self):
        """Test normalized embedding extraction and similarity math."""
        img_a = np.ones((128, 128, 3), dtype=np.uint8) * 128
        img_b = np.ones((128, 128, 3), dtype=np.uint8) * 128
        img_c = np.zeros((128, 128, 3), dtype=np.uint8)

        emb_a = self.embedder.extract_embedding(img_a)
        emb_b = self.embedder.extract_embedding(img_b)
        emb_c = self.embedder.extract_embedding(img_c)

        self.assertEqual(len(emb_a.shape), 1)
        sim_identical = self.embedder.compute_similarity(emb_a, emb_b)
        self.assertAlmostEqual(sim_identical, 1.0, places=2)

    def test_face_detector_no_face(self):
        """Test blank image results in no face detected."""
        blank_bytes = generate_blank_image()
        nparr = np.frombuffer(blank_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = self.detector.detect_faces(img, is_document=True)
        self.assertEqual(result.face_count, 0)
        self.assertFalse(result.is_usable)
        self.assertEqual(result.status, 'NO_FACE')

    def test_face_verification_service_execution(self):
        """Test full FaceVerificationService pipeline execution."""
        service = FaceVerificationService()
        result = service.process_verification(self.document)

        self.assertIn("document_face_detected", result)
        self.assertIn("selfie_face_detected", result)
        self.assertIn("similarity_score", result)
        self.assertIn("confidence", result)
        self.assertIn("match_status", result)
        self.assertIsInstance(result["similarity_score"], float)
        self.assertIsInstance(result["confidence"], float)

        # Ensure database model was populated
        face_record = FaceVerification.objects.get(document=self.document)
        self.assertIsNotNone(face_record)
        self.assertIn(face_record.match_status, [FaceVerification.MATCH_STATUS_MATCH, FaceVerification.MATCH_STATUS_MANUAL_REVIEW, FaceVerification.MATCH_STATUS_NO_FACE])

    def test_face_verification_missing_selfie_routes_to_manual_review(self):
        """Test that document without a selfie routes to MANUAL_REVIEW safely."""
        doc_no_selfie = Document.objects.create(
            verification_id="VX-2026-NOSELFIE",
            document_type=Document.DOC_TYPE_AADHAAR,
            original_file=self.doc_file,
            selfie_file=None,
            processing_status=Document.STATUS_UPLOADED
        )
        service = FaceVerificationService()
        result = service.process_verification(doc_no_selfie)

        self.assertFalse(result["selfie_face_detected"])
        self.assertEqual(result["similarity_score"], 0.0)
        self.assertIn(result["match_status"], ['NO_FACE', 'MANUAL_REVIEW'])

        doc_no_selfie.refresh_from_db()
        self.assertEqual(doc_no_selfie.processing_status, Document.STATUS_MANUAL_REVIEW)

    def test_api_face_verification_post_endpoint(self):
        """Test POST /api/face-verification/<verification_id>/ returns expected schema."""
        url = f"/api/face-verification/{self.document.verification_id}/"
        response = self.client.post(url, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("document_face_detected", data)
        self.assertIn("selfie_face_detected", data)
        self.assertIn("similarity_score", data)
        self.assertIn("confidence", data)
        self.assertIn("match_status", data)

    def test_api_face_verification_get_endpoint(self):
        """Test GET /api/face-verification/<verification_id>/ returns verification state."""
        # Run POST first
        self.client.post(f"/api/face-verification/{self.document.verification_id}/")

        response = self.client.get(f"/api/face-verification/{self.document.verification_id}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("match_status", data)

    def test_face_results_html_view(self):
        """Test HTML view /face/results/<verification_id>/ renders successfully."""
        response = self.client.get(reverse('face_verification:results', kwargs={'verification_id': self.document.verification_id}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "FACE VERIFICATION")
