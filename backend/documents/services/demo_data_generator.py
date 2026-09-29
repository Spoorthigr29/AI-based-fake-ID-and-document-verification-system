"""
VerifyX AI - Synthetic Test Case and Hackathon Demo Generator
============================================================
Generates 100% synthetic, non-PII identity document specimens and live selfie
artifacts demonstrating the 5 core forensic screening scenarios:

1. Case 1: NORMAL (Clean Authentic Specimen -> Low Risk)
2. Case 2: TAMPERED (Forensic Artifact & Splicing Anomaly -> High Tampering Risk)
3. Case 3: IDENTITY MISMATCH (Cross-field attribute conflict -> Manual Review)
4. Case 4: POOR QUALITY (Heavy Blur, Low Res, Skew -> Low Quality / Manual Review)
5. Case 5: FACE MISMATCH (Biometric Identity Discrepancy -> Manual Review)
"""

import io
import os
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.files.base import ContentFile

from documents.models import Document


def _draw_synthetic_face(draw, box, variant="face_a"):
    """Draws a stylized, synthetic avatar face within the specified bounding box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    # Head background
    draw.rectangle([x1, y1, x2, y2], fill=(220, 225, 230), outline=(100, 116, 139), width=2)
    
    if variant == "face_a":
        # Face oval (fair skin tone)
        draw.ellipse([x1 + w*0.2, y1 + h*0.15, x1 + w*0.8, y1 + h*0.85], fill=(245, 205, 175), outline=(180, 140, 110), width=2)
        # Hair
        draw.chord([x1 + w*0.15, y1 + h*0.08, x1 + w*0.85, y1 + h*0.45], start=180, end=360, fill=(40, 30, 20))
        # Eyes
        draw.ellipse([x1 + w*0.32, y1 + h*0.40, x1 + w*0.42, y1 + h*0.48], fill=(30, 41, 59))
        draw.ellipse([x1 + w*0.58, y1 + h*0.40, x1 + w*0.68, y1 + h*0.48], fill=(30, 41, 59))
        # Nose & Mouth
        draw.line([x1 + w*0.5, y1 + h*0.48, x1 + w*0.5, y1 + h*0.58], fill=(180, 130, 100), width=2)
        draw.arc([x1 + w*0.38, y1 + h*0.62, x1 + w*0.62, y1 + h*0.72], start=0, end=180, fill=(180, 80, 80), width=2)
    elif variant == "face_b":
        # Face rectangle / oval with deep bronze tone & completely different texture & dark sunglasses
        draw.ellipse([x1 + w*0.15, y1 + h*0.12, x1 + w*0.85, y1 + h*0.88], fill=(90, 50, 30), outline=(40, 20, 10), width=3)
        # Dark curly Afro hair
        draw.chord([x1 + w*0.10, y1 + h*0.02, x1 + w*0.90, y1 + h*0.42], start=180, end=360, fill=(10, 10, 10))
        # Large dark sunglasses covering top half of face (drastic LBP & gradient mismatch)
        draw.rectangle([x1 + w*0.20, y1 + h*0.35, x1 + w*0.48, y1 + h*0.52], fill=(15, 23, 42), outline=(255, 255, 255), width=2)
        draw.rectangle([x1 + w*0.52, y1 + h*0.35, x1 + w*0.80, y1 + h*0.52], fill=(15, 23, 42), outline=(255, 255, 255), width=2)
        draw.line([x1 + w*0.48, y1 + h*0.43, x1 + w*0.52, y1 + h*0.43], fill=(255, 255, 255), width=3)
        # Full thick beard
        draw.arc([x1 + w*0.22, y1 + h*0.55, x1 + w*0.78, y1 + h*0.88], start=0, end=180, fill=(10, 10, 10), width=8)


def generate_synthetic_case(case_name: str) -> dict:
    """
    Builds synthetic document and selfie images along with intake metadata
    for the requested hackathon demonstration case.

    Args:
        case_name: One of 'normal', 'tampered', 'identity_mismatch', 'poor_quality', 'face_mismatch'

    Returns:
        Dict with:
        - document_file: SimpleUploadedFile
        - selfie_file: SimpleUploadedFile (or None)
        - document_type: str
        - intake_data: dict
        - description: str
        - expected_outcome: dict
    """
    case_name = case_name.lower().strip()

    if case_name in ['normal', 'case_1', 'case1']:
        # CASE 1: NORMAL / AUTHENTIC
        img = Image.new('RGB', (1000, 620), color=(248, 250, 252))
        draw = ImageDraw.Draw(img)
        # Card Border
        draw.rounded_rectangle([15, 15, 985, 605], radius=16, outline=(30, 41, 59), width=4)
        # Header banner
        draw.rectangle([19, 19, 981, 100], fill=(224, 242, 254))
        draw.text((40, 40), "GOVERNMENT OF INDIA - SYNTHETIC SPECIMEN", fill=(3, 105, 161))
        # Photo box
        _draw_synthetic_face(draw, (50, 130, 250, 390), variant="face_a")
        # Field lines
        draw.text((280, 140), "Name: AARAV SHARMA", fill=(15, 23, 42))
        draw.text((280, 190), "DOB: 12/04/1992", fill=(15, 23, 42))
        draw.text((280, 240), "Gender: MALE", fill=(15, 23, 42))
        draw.text((280, 290), "Aadhaar No: 9876 5432 1098", fill=(15, 23, 42))
        draw.text((280, 340), "Address: 42 Silicon Lane, Indiranagar, Bengaluru 560038", fill=(15, 23, 42))
        draw.text((50, 420), "HELP-LINE: 1947 | WWW.UIDAI.GOV.IN (SYNTHETIC SAMPLE)", fill=(100, 116, 139))

        # Selfie Image (Matching Face A)
        selfie_img = Image.new('RGB', (500, 500), color=(241, 245, 249))
        s_draw = ImageDraw.Draw(selfie_img)
        _draw_synthetic_face(s_draw, (100, 80, 400, 420), variant="face_a")
        s_draw.text((30, 30), "LIVE APPLICANT WEBCAM CAPTURE", fill=(71, 85, 105))

        doc_buf = io.BytesIO()
        img.save(doc_buf, format='JPEG', quality=95)
        doc_buf.seek(0)

        selfie_buf = io.BytesIO()
        selfie_img.save(selfie_buf, format='JPEG', quality=95)
        selfie_buf.seek(0)

        return {
            "case_id": "case_1_normal",
            "title": "Case 1: Normal Authentic Document",
            "document_type": Document.DOC_TYPE_AADHAAR,
            "document_file": SimpleUploadedFile("case1_normal_aadhaar.jpg", doc_buf.getvalue(), content_type="image/jpeg"),
            "selfie_file": SimpleUploadedFile("case1_selfie.jpg", selfie_buf.getvalue(), content_type="image/jpeg"),
            "intake_data": {
                "name": "Aarav Sharma",
                "dob": "12/04/1992",
                "gender": "MALE",
                "id_number": "9876 5432 1098",
                "address": "42 Silicon Lane, Indiranagar, Bengaluru 560038"
            },
            "expected_outcome": {
                "risk_category": "LOW_RISK",
                "ocr_status": "HIGH",
                "face_status": "MATCH",
                "tampering_risk": "LOW",
                "consistency_status": "MATCH"
            },
            "description": "High-fidelity specimen with matching live biometric selfie, clear OCR text tokens, and zero tampering anomalies."
        }

    elif case_name in ['tampered', 'case_2', 'case2']:
        # CASE 2: TAMPERED FORENSIC SPECIMEN
        img = Image.new('RGB', (1000, 620), color=(248, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([15, 15, 985, 605], radius=16, outline=(30, 41, 59), width=4)
        draw.rectangle([19, 19, 981, 100], fill=(254, 226, 226))
        draw.text((40, 40), "INCOME TAX DEPARTMENT - PERMANENT ACCOUNT CARD", fill=(185, 28, 28))
        _draw_synthetic_face(draw, (50, 130, 250, 390), variant="face_a")
        draw.text((280, 140), "Name: ROHAN VERMA", fill=(15, 23, 42))
        draw.text((280, 190), "Father: SURESH VERMA", fill=(15, 23, 42))
        draw.text((280, 240), "DOB: 05/11/1988", fill=(15, 23, 42))
        draw.text((280, 290), "PAN: ABCPV9988K", fill=(15, 23, 42))

        # Introduce high-contrast digital tampering / spliced noise patches
        # 1. Pasted noise block over ID region with mismatched compression artifact
        for i in range(270, 520, 5):
            for j in range(280, 330, 5):
                noise_color = (random.randint(200, 255), random.randint(180, 230), random.randint(180, 230))
                draw.rectangle([i, j, i+4, j+4], fill=noise_color)
        draw.rectangle([270, 280, 520, 330], outline=(255, 0, 0), width=2)
        draw.text((285, 292), "PAN: ABCPV9988K (ALTERED)", fill=(0, 0, 0))

        # 2. Spliced photo boundary disturbance
        draw.rectangle([45, 125, 255, 395], outline=(239, 68, 68), width=3)

        # Save with heavy re-compression to trigger ELA
        doc_buf = io.BytesIO()
        img.save(doc_buf, format='JPEG', quality=75)
        doc_buf.seek(0)

        selfie_img = Image.new('RGB', (500, 500), color=(241, 245, 249))
        s_draw = ImageDraw.Draw(selfie_img)
        _draw_synthetic_face(s_draw, (100, 80, 400, 420), variant="face_a")
        selfie_buf = io.BytesIO()
        selfie_img.save(selfie_buf, format='JPEG', quality=95)
        selfie_buf.seek(0)

        return {
            "case_id": "case_2_tampered",
            "title": "Case 2: Digital Tampering & Spliced Region",
            "document_type": Document.DOC_TYPE_PAN,
            "document_file": SimpleUploadedFile("case2_tampered_pan.jpg", doc_buf.getvalue(), content_type="image/jpeg"),
            "selfie_file": SimpleUploadedFile("case2_selfie.jpg", selfie_buf.getvalue(), content_type="image/jpeg"),
            "intake_data": {
                "name": "ROHAN VERMA",
                "dob": "05/11/1988",
                "id_number": "ABCPV9988K"
            },
            "expected_outcome": {
                "risk_category": "HIGH_RISK",
                "tampering_risk": "HIGH",
                "ocr_status": "HIGH",
                "face_status": "MATCH"
            },
            "description": "Document exhibits compression variance and localized digital splicing over the document identifier field."
        }

    elif case_name in ['identity_mismatch', 'case_3', 'case3', 'mismatch']:
        # CASE 3: IDENTITY CROSS-FIELD MISMATCH
        img = Image.new('RGB', (1000, 620), color=(248, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([15, 15, 985, 605], radius=16, outline=(30, 41, 59), width=4)
        draw.rectangle([19, 19, 981, 100], fill=(224, 231, 255))
        draw.text((40, 40), "UNION OF INDIA - DRIVING LICENCE SPECIMEN", fill=(67, 56, 202))
        _draw_synthetic_face(draw, (50, 130, 250, 390), variant="face_a")
        draw.text((280, 140), "Name: PRIYA PATEL", fill=(15, 23, 42))
        draw.text((280, 190), "DOB: 15/08/1990", fill=(15, 23, 42))
        draw.text((280, 240), "DL No: DL-0420110023456", fill=(15, 23, 42))
        draw.text((280, 290), "Address: Flat 102, Green Meadows, Pune, Maharashtra", fill=(15, 23, 42))

        doc_buf = io.BytesIO()
        img.save(doc_buf, format='JPEG', quality=95)
        doc_buf.seek(0)

        selfie_img = Image.new('RGB', (500, 500), color=(241, 245, 249))
        s_draw = ImageDraw.Draw(selfie_img)
        _draw_synthetic_face(s_draw, (100, 80, 400, 420), variant="face_a")
        selfie_buf = io.BytesIO()
        selfie_img.save(selfie_buf, format='JPEG', quality=95)
        selfie_buf.seek(0)

        # Conflicting Applicant Intake Data
        return {
            "case_id": "case_3_identity_mismatch",
            "title": "Case 3: Cross-Field Identity Mismatch",
            "document_type": Document.DOC_TYPE_DRIVING_LICENSE,
            "document_file": SimpleUploadedFile("case3_mismatch_dl.jpg", doc_buf.getvalue(), content_type="image/jpeg"),
            "selfie_file": SimpleUploadedFile("case3_selfie.jpg", selfie_buf.getvalue(), content_type="image/jpeg"),
            "intake_data": {
                "name": "VIKRAM SINGH CHOUHAN",
                "dob": "01/01/1985",
                "id_number": "KA-0120199999999",
                "address": "Sector 4, Rohini, New Delhi 110085"
            },
            "expected_outcome": {
                "risk_category": "MANUAL_REVIEW",
                "consistency_status": "MISMATCH",
                "face_status": "MATCH",
                "tampering_risk": "LOW"
            },
            "description": "Applicant claim details (Name, DOB, ID number) conflict directly with the OCR-extracted document values."
        }

    elif case_name in ['poor_quality', 'case_4', 'case4', 'quality']:
        # CASE 4: POOR QUALITY / SEVERE BLUR / LOW RES
        base_img = Image.new('RGB', (300, 180), color=(200, 200, 200))
        draw = ImageDraw.Draw(base_img)
        draw.text((20, 20), "GOVT OF INDIA", fill=(80, 80, 80))
        draw.text((20, 60), "NAME: UNREADABLE", fill=(80, 80, 80))
        draw.text((20, 100), "DOB: **/**/****", fill=(80, 80, 80))
        
        # Heavy Gaussian Blur + Contrast Degradation + Rotation
        blurred = base_img.filter(ImageFilter.GaussianBlur(radius=3.5))
        enhancer = ImageEnhance.Contrast(blurred)
        degraded = enhancer.enhance(0.4)
        rotated = degraded.rotate(14, expand=True, fillcolor=(180, 180, 180))

        doc_buf = io.BytesIO()
        rotated.save(doc_buf, format='JPEG', quality=30)
        doc_buf.seek(0)

        return {
            "case_id": "case_4_poor_quality",
            "title": "Case 4: Severe Blur & Low Resolution",
            "document_type": Document.DOC_TYPE_AADHAAR,
            "document_file": SimpleUploadedFile("case4_poor_quality.jpg", doc_buf.getvalue(), content_type="image/jpeg"),
            "selfie_file": None,
            "intake_data": {
                "name": "SAMPLE USER",
                "dob": "10/10/1990"
            },
            "expected_outcome": {
                "risk_category": "MANUAL_REVIEW",
                "quality_status": "LOW_QUALITY"
            },
            "description": "Document image resolution and sharpness fall below forensic minimums, triggering physical document verification."
        }

    elif case_name in ['face_mismatch', 'case_5', 'case5', 'biometric']:
        # CASE 5: BIOMETRIC FACE MISMATCH
        img = Image.new('RGB', (1000, 620), color=(248, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([15, 15, 985, 605], radius=16, outline=(30, 41, 59), width=4)
        draw.rectangle([19, 19, 981, 100], fill=(254, 243, 199))
        draw.text((40, 40), "REPUBLIC OF INDIA - PASSPORT SPECIMEN", fill=(180, 83, 9))
        # Doc has Face A
        _draw_synthetic_face(draw, (50, 130, 250, 390), variant="face_a")
        draw.text((280, 140), "Name: ANANYA DESHMUKH", fill=(15, 23, 42))
        draw.text((280, 190), "Nationality: INDIAN", fill=(15, 23, 42))
        draw.text((280, 240), "DOB: 22/07/1995", fill=(15, 23, 42))
        draw.text((280, 290), "Passport No: Z9182734", fill=(15, 23, 42))

        doc_buf = io.BytesIO()
        img.save(doc_buf, format='JPEG', quality=95)
        doc_buf.seek(0)

        # Selfie has Face B (Completely Different Person)
        selfie_img = Image.new('RGB', (500, 500), color=(241, 245, 249))
        s_draw = ImageDraw.Draw(selfie_img)
        _draw_synthetic_face(s_draw, (100, 80, 400, 420), variant="face_b")
        s_draw.text((30, 30), "LIVE APPLICANT WEBCAM CAPTURE", fill=(71, 85, 105))

        selfie_buf = io.BytesIO()
        selfie_img.save(selfie_buf, format='JPEG', quality=95)
        selfie_buf.seek(0)

        return {
            "case_id": "case_5_face_mismatch",
            "title": "Case 5: Biometric Face Discrepancy",
            "document_type": Document.DOC_TYPE_PASSPORT,
            "document_file": SimpleUploadedFile("case5_doc_passport.jpg", doc_buf.getvalue(), content_type="image/jpeg"),
            "selfie_file": SimpleUploadedFile("case5_selfie_mismatch.jpg", selfie_buf.getvalue(), content_type="image/jpeg"),
            "intake_data": {
                "name": "ANANYA DESHMUKH",
                "dob": "22/07/1995",
                "id_number": "Z9182734"
            },
            "expected_outcome": {
                "risk_category": "MANUAL_REVIEW",
                "face_status": "MISMATCH"
            },
            "description": "Document photo and live webcam selfie represent two distinct facial topologies, failing automated biometric matching."
        }

    else:
        raise ValueError(f"Unknown synthetic case '{case_name}'. Choose from: normal, tampered, identity_mismatch, poor_quality, face_mismatch")
