import os
import sys
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from documents.models import Document
from face_verification.models import FaceVerification
from verification.services.verification_pipeline import VerificationPipelineOrchestrator
from documents.services.demo_data_generator import _draw_synthetic_face


def make_doc_image(face_variant='face_a', no_face=False):
    img = Image.new('RGB', (1000, 620), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([15, 15, 985, 605], radius=16, outline=(30, 41, 59), width=4)
    draw.rectangle([19, 19, 981, 100], fill=(224, 242, 254))
    draw.text((40, 40), 'GOVERNMENT OF INDIA - IDENTITY SPECIMEN', fill=(3, 105, 161))
    if not no_face:
        _draw_synthetic_face(draw, (50, 130, 250, 390), variant=face_variant)
    draw.text((280, 140), 'Name: AARAV SHARMA', fill=(15, 23, 42))
    draw.text((280, 190), 'DOB: 12/04/1992', fill=(15, 23, 42))
    draw.text((280, 240), 'Gender: MALE', fill=(15, 23, 42))
    draw.text((280, 290), 'Aadhaar No: 9876 5432 1098', fill=(15, 23, 42))
    draw.text((280, 340), 'Address: 42 Silicon Lane, Indiranagar, Bengaluru 560038', fill=(15, 23, 42))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    return SimpleUploadedFile('test_doc.jpg', buf.getvalue(), content_type='image/jpeg')


def make_selfie_image(face_variant='face_a', poor_quality=False):
    if poor_quality:
        img = Image.new('RGB', (120, 120), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        _draw_synthetic_face(draw, (20, 20, 100, 100), variant=face_variant)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=8.0))
        buf = io.BytesIO()
        blurred.save(buf, format='JPEG', quality=20)
        buf.seek(0)
        return SimpleUploadedFile('test_poor_selfie.jpg', buf.getvalue(), content_type='image/jpeg')
    else:
        img = Image.new('RGB', (500, 500), color=(241, 245, 249))
        draw = ImageDraw.Draw(img)
        _draw_synthetic_face(draw, (100, 80, 400, 420), variant=face_variant)
        draw.text((30, 30), 'LIVE APPLICANT WEBCAM CAPTURE', fill=(71, 85, 105))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        buf.seek(0)
        return SimpleUploadedFile('test_selfie.jpg', buf.getvalue(), content_type='image/jpeg')


def run_tests():
    orchestrator = VerificationPipelineOrchestrator()

    test_cases = [
        ('TEST 1 (Person A + Selfie Person A)', 'face_a', 'face_a', False, False, False),
        ('TEST 2 (Person A + Selfie Person B)', 'face_a', 'face_b', False, False, False),
        ('TEST 3 (Person A + No Selfie)', 'face_a', None, False, True, False),
        ('TEST 4 (No Portrait Doc + Selfie)', None, 'face_a', True, False, False),
        ('TEST 5 (Same Person + Poor Selfie)', 'face_a', 'face_a', False, False, True),
    ]

    print('\n' + '=' * 115)
    print(f'| {"Test Case":36s} | {"Doc Person":10s} | {"Selfie":12s} | {"Similarity":10s} | {"Liveness":12s} | {"Face Result":16s} | {"Risk Category":13s} |')
    print('=' * 115)

    for name, doc_var, self_var, no_doc_face, no_selfie, poor_selfie in test_cases:
        doc_file = make_doc_image(doc_var or 'face_a', no_face=no_doc_face)
        selfie_file = None if no_selfie else make_selfie_image(self_var or 'face_a', poor_quality=poor_selfie)
        
        doc = Document.objects.create(
            document_type=Document.DOC_TYPE_AADHAAR,
            original_file=doc_file,
            selfie_file=selfie_file,
            processing_status=Document.STATUS_UPLOADED
        )
        
        res = orchestrator.run_pipeline(doc)
        doc.refresh_from_db()
        fv = getattr(doc, 'face_verification', None)
        
        sim_str = f"{fv.similarity_score:.2f}" if (fv and fv.similarity_score is not None) else 'N/A'
        live_str = fv.liveness_status if fv else 'N/A'
        match_str = fv.match_status if fv else 'N/A'
        risk_cat = res.get('category', 'N/A')
        
        p_doc = 'None' if no_doc_face else (doc_var.upper() if doc_var else 'N/A')
        p_self = 'None' if no_selfie else (('POOR ' + self_var.upper()) if poor_selfie else self_var.upper())
        
        print(f'| {name:36s} | {p_doc:10s} | {p_self:12s} | {sim_str:10s} | {live_str:12s} | {match_str:16s} | {risk_cat:13s} |')

    print('=' * 115 + '\n')


if __name__ == '__main__':
    run_tests()
