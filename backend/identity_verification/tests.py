import io
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from documents.models import Document
from ocr_engine.models import OCRAnalysis, ExtractedField
from identity_verification.models import IdentityConsistencyCheck, VerificationResult
from identity_verification.services.normalization import (
    normalize_name,
    normalize_date,
    normalize_address,
    normalize_gender,
    normalize_doc_number,
)
from identity_verification.services.field_comparator import FieldComparator, FieldComparisonResult
from identity_verification.services.consistency_engine import IdentityConsistencyEngine

def generate_test_doc_image():
    img = Image.new('RGB', (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "NAME: RAHUL KUMAR", fill=(0, 0, 0))
    draw.text((40, 80), "DOB: 15/08/1995", fill=(0, 0, 0))
    draw.text((40, 120), "DOC ID: 1234 5678 9012", fill=(0, 0, 0))
    draw.text((40, 160), "ADDRESS: 123 Main St, Apt 4B, New Delhi 110001", fill=(0, 0, 0))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='JPEG')
    byte_arr.seek(0)
    return byte_arr.getvalue()

class IdentityConsistencyTests(TestCase):
    """Test suite for field normalization, fuzzy comparisons, consistency engine and REST API."""

    def setUp(self):
        self.client = Client()
        self.comparator = FieldComparator()
        self.engine = IdentityConsistencyEngine()

        doc_bytes = generate_test_doc_image()
        self.document = Document.objects.create(
            verification_id="VX-2026-CONSIST01",
            original_file=SimpleUploadedFile("doc.jpg", doc_bytes, content_type="image/jpeg"),
            file_hash="test_hash_consistency_123",
            processing_status=Document.STATUS_UPLOADED
        )

        # Pre-seed OCR analysis fields
        ocr_analysis = OCRAnalysis.objects.create(
            document=self.document,
            raw_text="NAME: RAHUL KUMAR DOB: 15/08/1995 DOC ID: 1234 5678 9012 ADDRESS: 123 Main St, Apt 4B, New Delhi 110001",
            overall_confidence=0.96
        )
        ExtractedField.objects.create(document=self.document, field_name="name", field_value="RAHUL KUMAR", confidence=0.96)
        ExtractedField.objects.create(document=self.document, field_name="date_of_birth", field_value="15/08/1995", confidence=0.95)
        ExtractedField.objects.create(document=self.document, field_name="document_number", field_value="1234 5678 9012", confidence=0.97)
        ExtractedField.objects.create(document=self.document, field_name="address", field_value="123 Main St, Apt 4B, New Delhi 110001", confidence=0.92)

    def test_name_normalization(self):
        """Test name normalization rules (case folding, honorific stripping, space collapsing)."""
        self.assertEqual(normalize_name("RAHUL KUMAR"), "rahul kumar")
        self.assertEqual(normalize_name("Rahul Kumar"), "rahul kumar")
        self.assertEqual(normalize_name("rahul kumar"), "rahul kumar")
        self.assertEqual(normalize_name("Mr. Rahul   Kumar"), "rahul kumar")
        self.assertEqual(normalize_name("Dr. Rahul Kumar, Ph.D"), "rahul kumar phd")
        self.assertEqual(normalize_name("Shri Rahul Kumar"), "rahul kumar")

    def test_date_normalization(self):
        """Test date parsing and ISO normalization across diverse formats."""
        self.assertEqual(normalize_date("1995-08-15"), "1995-08-15")
        self.assertEqual(normalize_date("15/08/1995"), "1995-08-15")
        self.assertEqual(normalize_date("15-08-1995"), "1995-08-15")
        self.assertEqual(normalize_date("15.08.1995"), "1995-08-15")
        self.assertEqual(normalize_date("15 August 1995"), "1995-08-15")
        self.assertEqual(normalize_date("Aug 15, 1995"), "1995-08-15")

    def test_address_normalization(self):
        """Test address normalization with abbreviation standardization."""
        addr1 = normalize_address("123 Main St., Apt. 4B, New Delhi")
        self.assertIn("street", addr1)
        self.assertIn("apartment", addr1)
        self.assertIn("123 main street apartment 4b new delhi", addr1)

    def test_fuzzy_name_matching(self):
        """Test fuzzy name matching for minor typos and token permutations."""
        res_exact = self.comparator.compare_names("Rahul Kumar", "RAHUL KUMAR")
        self.assertTrue(res_exact.is_match)
        self.assertEqual(res_exact.similarity_score, 1.0)

        res_perm = self.comparator.compare_names("Rahul Kumar", "Kumar Rahul")
        self.assertTrue(res_perm.is_match)
        self.assertGreaterEqual(res_perm.similarity_score, 0.95)

        res_typo = self.comparator.compare_names("Rahul Kumar", "Rahul Kumaar")
        self.assertTrue(res_typo.is_match)
        self.assertGreaterEqual(res_typo.similarity_score, 0.90)

        res_diff = self.comparator.compare_names("Rahul Kumar", "Amit Sharma")
        self.assertFalse(res_diff.is_match)
        self.assertLess(res_diff.similarity_score, 0.50)

    def test_consistency_engine_with_address_mismatch(self):
        """Test consistency engine scoring when name, DOB, ID match but address differs."""
        claims = {
            "name": "Rahul Kumar",
            "date_of_birth": "1995-08-15",
            "document_number": "1234 5678 9012",
            "address": "456 Different Street, Bangalore 560001",  # Mismatched address
            "gender": "MALE"
        }

        result = self.engine.process_consistency(self.document, applicant_claims=claims)

        self.assertTrue(result["name_match"])
        self.assertTrue(result["dob_match"])
        self.assertTrue(result["document_number_match"])
        self.assertFalse(result["address_match"])
        self.assertGreaterEqual(result["consistency_score"], 80)
        self.assertLess(result["consistency_score"], 90)

        # Confirm DB persistence
        record = IdentityConsistencyCheck.objects.get(document=self.document)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, IdentityConsistencyCheck.STATUS_MANUAL_REVIEW)
        self.assertIn("⚠ Address mismatch", record.checklist)

    def test_consistency_api_post_endpoint(self):
        """Test POST /api/verify/consistency/<verification_id>/ returns expected schema."""
        url = f"/api/verify/consistency/{self.document.verification_id}/"
        payload = {
            "claims": {
                "name": "Rahul Kumar",
                "date_of_birth": "1995-08-15",
                "document_number": "1234 5678 9012",
                "address": "999 Alternative Rd, Chennai 600001"
            }
        }
        response = self.client.post(url, data=payload, content_type="application/json")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("name_match", data)
        self.assertIn("dob_match", data)
        self.assertIn("document_number_match", data)
        self.assertIn("address_match", data)
        self.assertIn("consistency_score", data)
        self.assertIsInstance(data["consistency_score"], int)

    def test_consistency_api_get_endpoint(self):
        """Test GET /api/verify/consistency/<verification_id>/ returns stored state."""
        self.client.post(f"/api/verify/consistency/{self.document.verification_id}/", data={}, content_type="application/json")
        response = self.client.get(f"/api/verify/consistency/{self.document.verification_id}/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("consistency_score", data)
        self.assertIn("status", data)

    def test_consistency_results_html_view(self):
        """Test HTML view /verify/results/<verification_id>/ renders without error."""
        response = self.client.get(reverse('identity_verification:results', kwargs={'verification_id': self.document.verification_id}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Identity Consistency")
