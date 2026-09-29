import os
import sys
import io
import django
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from documents.validators import validate_uploaded_document
from documents.models import Document
from ocr_engine.services.ocr_service import OCREngineService
from ocr_engine.services.preprocessing import ImagePreprocessor
from PIL import Image
import pypdf

def test_validators():
    print("Testing validators...")
    
    # 1. Valid PDF file
    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n188\n%%EOF"
    valid_pdf = SimpleUploadedFile("sample.pdf", pdf_bytes, content_type="application/pdf")
    validate_uploaded_document(valid_pdf)
    print("[OK] Valid PDF validator passed")
    
    # 2. Oversized file (>10MB)
    large_content = b"%PDF-1.4" + (b"x" * (10 * 1024 * 1024 + 10))
    oversized_file = SimpleUploadedFile("large.pdf", large_content, content_type="application/pdf")
    try:
        validate_uploaded_document(oversized_file)
        assert False, "Oversized file should have failed validation"
    except ValidationError as e:
        assert "File size exceeds the 10 MB limit. Please upload a smaller document." in str(e)
        print(f"[OK] Oversized error message matched: {e}")

    # 3. Unsupported extension (.exe, .txt, .docx)
    invalid_ext_file = SimpleUploadedFile("test.exe", b"MZ\x90\x00", content_type="application/octet-stream")
    try:
        validate_uploaded_document(invalid_ext_file)
        assert False, "Invalid extension should have failed validation"
    except ValidationError as e:
        assert "Unsupported file format. Please upload PDF, JPG, JPEG, PNG or WEBP." in str(e)
        print(f"[OK] Unsupported format error message matched: {e}")

def test_pdf_image_rendering():
    print("Testing PDF rendering for OCR/Face detection...")
    # Create a real single-page PDF with PIL image
    img = Image.new("RGB", (300, 200), color=(240, 240, 240))
    pdf_io = io.BytesIO()
    img.save(pdf_io, format="PDF")
    pdf_bytes = pdf_io.getvalue()
    
    rendered = ImagePreprocessor.load_image(pdf_bytes)
    assert rendered is not None
    assert rendered.shape[0] > 0 and rendered.shape[1] > 0
    print(f"[OK] PDF successfully rendered to BGR numpy image: shape={rendered.shape}")
    
    # Test OCR service on PDF bytes
    ocr_result = OCREngineService.process_document(pdf_bytes)
    assert ocr_result is not None
    print(f"[OK] OCR service successfully processed PDF. Keys: {list(ocr_result.keys())}")

if __name__ == '__main__':
    test_validators()
    test_pdf_image_rendering()
    print("\nALL PDF TESTS PASSED SUCCESSFULLY!")
