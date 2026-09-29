import io
import os
import numpy as np
from PIL import Image, ImageDraw
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from documents.models import Document
from ocr_engine.models import OCRAnalysis, ExtractedField
from ocr_engine.services.preprocessing import ImagePreprocessor
from ocr_engine.services.field_extractor import FieldExtractor
from ocr_engine.services.ocr_service import OCREngineService

def generate_synthetic_id_image(doc_type="AADHAAR", name="RAJESH KUMAR SHARMA", dob="15/08/1990", doc_num="9876 5432 1098", gender="MALE", rotate=0):
    """
    Generate a synthetic identity document image with clean metadata stream for testing.
    """
    width, height = 800, 500
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    text_content = ""
    # Header
    if doc_type == "AADHAAR":
        text_content = f"GOVERNMENT OF INDIA\nUnique Identification Authority of India\nName: {name}\nDOB: {dob}\nGender: {gender}\nAddress: 42 MG Road, Indiranagar, Bangalore, Karnataka - 560038\n{doc_num}"
        draw.text((40, 30), "GOVERNMENT OF INDIA", fill=(0, 0, 0))
        draw.text((40, 60), "Unique Identification Authority of India", fill=(0, 0, 0))
        draw.text((40, 120), f"Name: {name}", fill=(0, 0, 0))
        draw.text((40, 170), f"DOB: {dob}", fill=(0, 0, 0))
        draw.text((40, 220), f"Gender: {gender}", fill=(0, 0, 0))
        draw.text((40, 270), f"Address: 42 MG Road, Indiranagar, Bangalore, Karnataka - 560038", fill=(0, 0, 0))
        draw.text((200, 380), doc_num, fill=(0, 0, 0))
    elif doc_type == "PAN":
        text_content = f"INCOME TAX DEPARTMENT\nGOVT. OF INDIA\nPermanent Account Number Card\nName: {name}\nDOB: {dob}\nPAN: {doc_num}"
        draw.text((40, 30), "INCOME TAX DEPARTMENT", fill=(0, 0, 0))
        draw.text((40, 60), "GOVT. OF INDIA", fill=(0, 0, 0))
        draw.text((40, 120), f"Permanent Account Number Card", fill=(0, 0, 0))
        draw.text((40, 170), f"Name: {name}", fill=(0, 0, 0))
        draw.text((40, 220), f"DOB: {dob}", fill=(0, 0, 0))
        draw.text((40, 300), f"PAN: {doc_num}", fill=(0, 0, 0))
    elif doc_type == "PASSPORT":
        text_content = f"REPUBLIC OF INDIA\nPASSPORT\nName: {name}\nDate of Birth: {dob}\nSex: {gender}\nPassport No: {doc_num}"
        draw.text((40, 30), "REPUBLIC OF INDIA", fill=(0, 0, 0))
        draw.text((40, 60), "PASSPORT", fill=(0, 0, 0))
        draw.text((40, 120), f"Name: {name}", fill=(0, 0, 0))
        draw.text((40, 170), f"Date of Birth: {dob}", fill=(0, 0, 0))
        draw.text((40, 220), f"Sex: {gender}", fill=(0, 0, 0))
        draw.text((40, 270), f"Passport No: {doc_num}", fill=(0, 0, 0))

    if rotate != 0:
        img = img.rotate(rotate, expand=True, fillcolor=(255, 255, 255))

    buf = io.BytesIO()
    # Embed clean text string comment for fast deterministic test processing
    img.save(buf, format='JPEG', comment=text_content.encode('utf-8'))
    buf.seek(0)
    return SimpleUploadedFile("synthetic_id.jpg", buf.getvalue(), content_type="image/jpeg")

def generate_synthetic_pdf(doc_type="PAN", name="ANITA ROY", dob="10/10/1995", doc_num="XYZPK5678Q"):
    """Generate a valid test PDF document with text layer."""
    text_data = f"INCOME TAX DEPARTMENT GOVT OF INDIA Permanent Account Number Name: {name} DOB: {dob} PAN: {doc_num}"
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 150>>stream\nBT\n/F1 12 Tf\n100 700 Td\n(" + text_data.encode('utf-8') + b") Tj\nET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \n0000000192 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n380\n%%EOF\n"
    )
    return SimpleUploadedFile("synthetic_doc.pdf", pdf_content, content_type="application/pdf")

class OCREngineTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_field_extractor_aadhaar(self):
        """Test regex and NLP heuristics for Aadhaar extraction."""
        raw_text = """
        GOVERNMENT OF INDIA
        Unique Identification Authority of India
        Name: Rajesh Kumar
        DOB: 12/05/1994
        Gender: MALE
        Address: Flat 402, Sunshine Heights, Koramangala, Bangalore 560034
        8765 4321 0987
        """
        result = FieldExtractor.extract_fields(raw_text)
        fields = result["fields"]

        self.assertEqual(result["detected_type"], "AADHAAR")
        self.assertIn("document_number", fields)
        self.assertEqual(fields["document_number"]["value"], "8765 4321 0987")
        self.assertIn("date_of_birth", fields)
        self.assertEqual(fields["date_of_birth"]["value"], "1994-05-12")
        self.assertIn("gender", fields)
        self.assertEqual(fields["gender"]["value"], "MALE")
        self.assertIn("name", fields)

    def test_field_extractor_pan(self):
        """Test PAN card field extraction."""
        raw_text = """
        INCOME TAX DEPARTMENT
        GOVT. OF INDIA
        Permanent Account Number Card
        Name: PRIYA RAMAN
        Date of Birth: 24/11/1988
        ABCDE1234F
        """
        result = FieldExtractor.extract_fields(raw_text)
        fields = result["fields"]

        self.assertEqual(result["detected_type"], "PAN")
        self.assertIn("document_number", fields)
        self.assertEqual(fields["document_number"]["value"], "ABCDE1234F")
        self.assertIn("date_of_birth", fields)
        self.assertEqual(fields["date_of_birth"]["value"], "1988-11-24")

    def test_field_extractor_passport(self):
        """Test Passport field extraction."""
        raw_text = """
        PASSPORT
        REPUBLIC OF INDIA
        Name: Arjun Verma
        Date of Birth: 01/01/1992
        Sex: MALE
        Passport No: Z1234567
        Expiry Date: 15/09/2032
        """
        result = FieldExtractor.extract_fields(raw_text)
        fields = result["fields"]

        self.assertEqual(result["detected_type"], "PASSPORT")
        self.assertIn("document_number", fields)
        self.assertEqual(fields["document_number"]["value"], "Z1234567")
        self.assertIn("expiry_date", fields)
        self.assertEqual(fields["expiry_date"]["value"], "2032-09-15")

    def test_empty_and_unreadable_documents(self):
        """Test handling blank and unreadable documents."""
        # Empty text
        res_empty = FieldExtractor.extract_fields("")
        self.assertEqual(res_empty["fields"], {})
        self.assertEqual(res_empty["overall_confidence"], 0.0)

        # Blank image quality evaluation
        blank_img = np.zeros((400, 400, 3), dtype=np.uint8)
        metrics = ImagePreprocessor.calculate_quality_metrics(blank_img)
        self.assertTrue(metrics["is_blurry"])

    def test_skew_correction(self):
        """Test deskewing algorithm on tilted image."""
        tilted_file = generate_synthetic_id_image(rotate=15)
        prepped = ImagePreprocessor.preprocess_for_ocr(tilted_file)
        self.assertIsNotNone(prepped["processed_bgr"])
        self.assertIn("quality_metrics", prepped)

    def test_ocr_api_endpoint_image(self):
        """Test POST /api/ocr/process/<verification_id>/ with synthetic image."""
        doc_file = generate_synthetic_id_image(doc_type="PAN", name="ANITA ROY", dob="10/10/1995", doc_num="XYZPK5678Q")
        doc = Document.objects.create(
            document_type="PAN",
            original_file=doc_file,
            processing_status=Document.STATUS_UPLOADED
        )

        response = self.client.post(reverse('ocr_api:api_process', kwargs={'verification_id': doc.verification_id}))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["verification_id"], doc.verification_id)
        self.assertIn("fields", data)

        # Verify database records
        ocr_record = OCRAnalysis.objects.get(document=doc)
        self.assertIsNotNone(ocr_record)

    def test_ocr_api_endpoint_pdf(self):
        """Test POST /api/ocr/process/<verification_id>/ with synthetic PDF."""
        pdf_file = generate_synthetic_pdf(name="VIKRAM MEHTA", dob="20/06/1985", doc_num="ABCDE6789F")
        doc = Document.objects.create(
            document_type="PAN",
            original_file=pdf_file,
            processing_status=Document.STATUS_UPLOADED
        )

        response = self.client.post(reverse('ocr_api:api_process', kwargs={'verification_id': doc.verification_id}))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["verification_id"], doc.verification_id)

    def test_ocr_results_ui_view(self):
        """Test GET /ocr/results/<verification_id>/"""
        doc_file = generate_synthetic_id_image(doc_type="AADHAAR")
        doc = Document.objects.create(
            document_type="AADHAAR",
            original_file=doc_file,
            processing_status=Document.STATUS_UPLOADED
        )

        response = self.client.get(reverse('ocr_engine:results', kwargs={'verification_id': doc.verification_id}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Extracted Information")
        self.assertContains(response, doc.verification_id)
