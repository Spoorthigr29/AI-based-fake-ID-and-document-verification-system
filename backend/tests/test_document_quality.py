import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from documents.models import Document, DocumentQualityAnalysis
from documents.services.quality_analyzer import DocumentQualityAnalyzer
from documents.services.document_classifier import DocumentClassifier

def create_synthetic_card(doc_type="sample_identity_card", width=800, height=500, blur=False, dark=False):
    """Generate in-memory synthetic card with customizable artifacts."""
    color = (30, 30, 30) if dark else (255, 255, 255)
    text_color = (180, 180, 180) if dark else (0, 0, 0)
    
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)

    text_snippet = ""
    if doc_type == "sample_identity_card":
        text_snippet = "GOVERNMENT OF INDIA UNIQUE IDENTIFICATION AADHAAR 9988 7766 5544"
        draw.text((30, 30), "GOVERNMENT OF INDIA", fill=text_color)
        draw.text((30, 70), "Unique Identification Authority of India", fill=text_color)
        draw.text((30, 120), "Name: SAMPLE USER", fill=text_color)
        draw.text((30, 160), "9988 7766 5544", fill=text_color)
    elif doc_type == "sample_pan_document":
        text_snippet = "INCOME TAX DEPARTMENT GOVT OF INDIA PAN CARD ABCDE1234F"
        draw.text((30, 30), "INCOME TAX DEPARTMENT", fill=text_color)
        draw.text((30, 70), "Permanent Account Number Card", fill=text_color)
        draw.text((30, 120), "PAN: ABCDE1234F", fill=text_color)
    elif doc_type == "sample_passport":
        text_snippet = "REPUBLIC OF INDIA PASSPORT P<IND J1234567"
        draw.text((30, 30), "REPUBLIC OF INDIA", fill=text_color)
        draw.text((30, 70), "PASSPORT", fill=text_color)
        draw.text((30, 120), "Passport No: J1234567", fill=text_color)
    elif doc_type == "sample_driving_license":
        text_snippet = "UNION OF INDIA DRIVING LICENCE TRANSPORT DL NO DL0420110012345"
        draw.text((30, 30), "UNION OF INDIA DRIVING LICENCE", fill=text_color)
        draw.text((30, 70), "DL NO: DL0420110012345", fill=text_color)
    else:
        text_snippet = "GENERIC SAMPLE NOTE UNKNOWN DOCUMENT"
        draw.text((30, 30), "GENERIC UNKNOWN NOTE", fill=text_color)

    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=8))

    buf = io.BytesIO()
    img.save(buf, format='JPEG', comment=text_snippet.encode('utf-8'))
    buf.seek(0)
    return SimpleUploadedFile("synthetic_card.jpg", buf.getvalue(), content_type="image/jpeg")

class DocumentQualityAndClassificationTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_classifier_categories(self):
        """Test classification for all 5 document categories."""
        # 1. Identity Card
        res_id = DocumentClassifier.classify("dummy.jpg", ocr_text="GOVERNMENT OF INDIA UNIQUE IDENTIFICATION 1234 5678 9012")
        self.assertEqual(res_id["document_type"], "AADHAAR")

        # 2. PAN Card
        res_pan = DocumentClassifier.classify("dummy.jpg", ocr_text="INCOME TAX DEPARTMENT PERMANENT ACCOUNT NUMBER ABCDE1234F")
        self.assertEqual(res_pan["document_type"], "PAN")

        # 3. Passport
        res_pass = DocumentClassifier.classify("dummy.jpg", ocr_text="REPUBLIC OF INDIA PASSPORT P<IND Z1234567")
        self.assertEqual(res_pass["document_type"], "PASSPORT")

        # 4. Driving License
        res_dl = DocumentClassifier.classify("dummy.jpg", ocr_text="UNION OF INDIA DRIVING LICENCE TRANSPORT DL NO DL1234567890123")
        self.assertEqual(res_dl["document_type"], "DRIVING_LICENSE")

        # 5. Unknown
        res_unk = DocumentClassifier.classify("dummy.jpg", ocr_text="RANDOM TEXT NOTE WITHOUT RECOGNIZED MARKERS")
        self.assertEqual(res_unk["document_type"], "OTHER")

    def test_quality_analysis_high_quality(self):
        """Test quality analysis on a crisp, standard-sized document."""
        good_card = create_synthetic_card("sample_identity_card", width=1000, height=600)
        analysis = DocumentQualityAnalyzer.analyze_quality(good_card)

        self.assertGreaterEqual(analysis["quality_score"], 70)
        self.assertFalse(analysis["is_poor_quality"])
        self.assertEqual(analysis["recommended_status"], "UPLOADED")
        self.assertIn("✓ Resolution acceptable", analysis["checklist"])

    def test_quality_analysis_blurry_document(self):
        """Test quality analyzer detects severe blur and flags for manual review."""
        blurry_card = create_synthetic_card("sample_identity_card", blur=True)
        analysis = DocumentQualityAnalyzer.analyze_quality(blurry_card)

        self.assertTrue(analysis["is_poor_quality"])
        self.assertEqual(analysis["recommended_status"], "MANUAL_REVIEW")
        self.assertTrue(any("blur" in issue.lower() for issue in analysis["issues"]))

    def test_quality_analysis_dark_document(self):
        """Test quality analyzer detects underexposure and flags for manual review."""
        dark_card = create_synthetic_card("sample_pan_document", dark=True)
        analysis = DocumentQualityAnalyzer.analyze_quality(dark_card)

        self.assertTrue(analysis["is_poor_quality"])
        self.assertTrue(any("underexposed" in issue.lower() or "dark" in issue.lower() or "lighting" in issue.lower() for issue in analysis["issues"]))

    def test_analyze_api_endpoint(self):
        """Test POST /api/documents/analyze/<verification_id>/"""
        card_file = create_synthetic_card("sample_identity_card")
        doc = Document.objects.create(
            document_type=Document.DOC_TYPE_AADHAAR,
            original_file=card_file,
            processing_status=Document.STATUS_UPLOADED
        )

        response = self.client.post(reverse('documents_api:api_analyze', kwargs={'verification_id': doc.verification_id}))
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate exact response format
        self.assertIn("document_type", data)
        self.assertEqual(data["document_type"], "AADHAAR")
        self.assertIn("quality_score", data)
        self.assertIsInstance(data["quality_score"], int)
        self.assertIn("issues", data)
        self.assertIsInstance(data["issues"], list)

        # Ensure database record was created
        qa_rec = DocumentQualityAnalysis.objects.get(document=doc)
        self.assertEqual(qa_rec.classified_type, "AADHAAR")

    def test_blurry_upload_routes_to_manual_review(self):
        """Test that uploading a severely blurry image automatically sets status to MANUAL_REVIEW."""
        blurry_file = create_synthetic_card("sample_pan_document", blur=True)
        response = self.client.post(reverse('documents_api:api_upload'), {
            'document_type': 'PAN',
            'original_file': blurry_file,
        })
        self.assertEqual(response.status_code, 201)
        ver_id = response.json()['verification_id']

        doc = Document.objects.get(verification_id=ver_id)
        self.assertEqual(doc.processing_status, Document.STATUS_MANUAL_REVIEW)
        self.assertTrue(doc.quality_analysis.is_poor_quality)
