import io
from PIL import Image
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from documents.models import Document
from documents.validators import MAX_FILE_SIZE_BYTES

def create_test_image(format='JPEG', size=(800, 500), color='white'):
    """Helper to generate in-memory valid image file with text."""
    from PIL import ImageDraw
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((40, 40), "VALID IDENTITY DOCUMENT SAMPLE 1234 5678 9012", fill=(0, 0, 0))
    draw.text((40, 80), "NAME: TEST APPLICANT DOB: 1990-01-01", fill=(0, 0, 0))
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=format)
    byte_arr.seek(0)
    ext = 'jpg' if format.upper() == 'JPEG' else format.lower()
    content_type = 'image/jpeg' if format.upper() == 'JPEG' else f'image/{ext}'
    return SimpleUploadedFile(f"test_file.{ext}", byte_arr.getvalue(), content_type=content_type)

def create_test_pdf():
    """Helper to generate a minimal valid in-memory PDF."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n"
    )
    return SimpleUploadedFile("test_doc.pdf", pdf_content, content_type='application/pdf')

class DocumentUploadTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_upload_valid_jpg(self):
        """Test uploading valid JPG document."""
        jpg_file = create_test_image('JPEG')
        response = self.client.post(reverse('verification:upload'), {
            'document_type': 'PASSPORT',
            'original_file': jpg_file,
        })
        self.assertEqual(response.status_code, 302)
        doc = Document.objects.first()
        self.assertIsNotNone(doc)
        self.assertTrue(doc.verification_id.startswith('VX-'))
        self.assertEqual(len(doc.file_hash), 64)
        self.assertEqual(doc.processing_status, Document.STATUS_UPLOADED)

    def test_upload_valid_png(self):
        """Test uploading valid PNG document with selfie."""
        png_doc = create_test_image('PNG')
        jpg_selfie = create_test_image('JPEG', color='green')
        response = self.client.post(reverse('verification:upload'), {
            'document_type': 'AADHAAR',
            'original_file': png_doc,
            'selfie_file': jpg_selfie,
        })
        self.assertEqual(response.status_code, 302)
        doc = Document.objects.first()
        self.assertIsNotNone(doc)
        self.assertTrue(doc.selfie_file.name != '')
        self.assertEqual(len(doc.selfie_hash), 64)

    def test_upload_valid_pdf(self):
        """Test uploading valid PDF document."""
        pdf_file = create_test_pdf()
        response = self.client.post(reverse('verification:upload'), {
            'document_type': 'PAN',
            'original_file': pdf_file,
        })
        self.assertEqual(response.status_code, 302)
        doc = Document.objects.first()
        self.assertIsNotNone(doc)
        self.assertTrue(doc.file_hash)

    def test_upload_oversized_file(self):
        """Test rejecting files larger than 10MB."""
        oversized_content = b"A" * (MAX_FILE_SIZE_BYTES + 1024)
        oversized_file = SimpleUploadedFile("big.jpg", oversized_content, content_type="image/jpeg")
        response = self.client.post(reverse('verification:upload'), {
            'document_type': 'PASSPORT',
            'original_file': oversized_file,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Document.objects.count(), 0)

    def test_upload_invalid_extension(self):
        """Test rejecting executable or unsupported file extensions."""
        bad_file = SimpleUploadedFile("malware.exe", b"MZ\x90\x00\x03", content_type="application/x-msdownload")
        response = self.client.post(reverse('verification:upload'), {
            'document_type': 'PASSPORT',
            'original_file': bad_file,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Document.objects.count(), 0)

    def test_upload_empty_file(self):
        """Test rejecting 0-byte files."""
        empty_file = SimpleUploadedFile("empty.jpg", b"", content_type="image/jpeg")
        response = self.client.post(reverse('verification:upload'), {
            'document_type': 'PASSPORT',
            'original_file': empty_file,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Document.objects.count(), 0)

    def test_upload_corrupted_image(self):
        """Test rejecting fake image with invalid payload."""
        fake_jpg = SimpleUploadedFile("fake.jpg", b"\xff\xd8\xff\xe0NOT_A_VALID_JPEG_STREAM", content_type="image/jpeg")
        response = self.client.post(reverse('verification:upload'), {
            'document_type': 'PASSPORT',
            'original_file': fake_jpg,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Document.objects.count(), 0)

    def test_api_upload_endpoint(self):
        """Test REST API endpoint POST /api/documents/upload/"""
        jpg_doc = create_test_image('JPEG')
        jpg_selfie = create_test_image('JPEG', color='red')
        response = self.client.post(reverse('documents_api:api_upload'), {
            'document_type': 'DRIVING_LICENSE',
            'original_file': jpg_doc,
            'selfie_file': jpg_selfie,
        }, format='multipart')

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertTrue(data['verification_id'].startswith('VX-'))
        self.assertEqual(data['message'], 'Document uploaded successfully')
