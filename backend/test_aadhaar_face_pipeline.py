import os
import sys
from pathlib import Path
import django
import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from documents.models import Document
from face_verification.services.face_service import FaceVerificationService
from documents.services.aadhaar_fixes import decode_aadhaar_qr
from risk_engine.services.risk_calculator import DeterministicRiskCalculator

face_service = FaceVerificationService()
risk_calc = DeterministicRiskCalculator()

def prepare_test_assets():
    """Ensure test assets have distinct genuine vs impostor and no-photo cases."""
    test_dir = 'backend/media/test_suite'
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Genuine Doc Aadhaar
    real_doc_p = 'backend/media/documents/2026/09/27/c77a3e27e14d4eeebebd6871709f2808.jpg'
    if os.path.exists(real_doc_p):
        doc_img = cv2.imread(real_doc_p)
    else:
        doc_img = cv2.imread(os.path.join(test_dir, 'doc_genuine_aadhaar.jpg'))
    
    # 2. Doc without photo (mask left section where portrait resides)
    doc_no_photo = doc_img.copy()
    h, w = doc_no_photo.shape[:2]
    doc_no_photo[:, :int(w*0.60)] = 245
    cv2.imwrite(os.path.join(test_dir, 'doc_no_photo.jpg'), doc_no_photo)

    # 3. Selfie Person B (Distinct Person with different skin tones/structure)
    # Generate an authentic distinct face image
    selfie_b = np.full((400, 400, 3), (220, 220, 220), dtype=np.uint8)
    # Draw distinct facial features (oval face with distinct tone and hair)
    cv2.ellipse(selfie_b, (200, 210), (105, 140), 0, 0, 360, (140, 175, 210), -1) # distinct warm tone
    cv2.ellipse(selfie_b, (200, 105), (110, 50), 0, 0, 360, (30, 25, 20), -1) # distinct dark hair
    # Eyes
    cv2.circle(selfie_b, (160, 185), 14, (255, 255, 255), -1)
    cv2.circle(selfie_b, (160, 185), 6, (40, 20, 10), -1)
    cv2.circle(selfie_b, (240, 185), 14, (255, 255, 255), -1)
    cv2.circle(selfie_b, (240, 185), 6, (40, 20, 10), -1)
    # Nose & Mouth
    cv2.line(selfie_b, (200, 195), (200, 230), (100, 130, 160), 3)
    cv2.ellipse(selfie_b, (200, 270), (35, 14), 0, 0, 360, (80, 90, 160), -1)
    cv2.imwrite(os.path.join(test_dir, 'selfie_person_b.jpg'), selfie_b)

def run_test_case(name, doc_path, selfie_path):
    print(f"\n==================================================")
    print(f"TEST: {name}")
    print(f"==================================================")
    
    with open(doc_path, 'rb') as f:
        doc_file = SimpleUploadedFile(os.path.basename(doc_path), f.read(), content_type='image/jpeg')
    
    selfie_file = None
    if selfie_path and os.path.exists(selfie_path):
        with open(selfie_path, 'rb') as f:
            selfie_file = SimpleUploadedFile(os.path.basename(selfie_path), f.read(), content_type='image/jpeg')
            
    doc = Document.objects.create(
        document_type=Document.DOC_TYPE_AADHAAR,
        original_file=doc_file,
        selfie_file=selfie_file,
        processing_status=Document.STATUS_PROCESSING
    )
    
    # 1. Face Verification Pipeline
    face_res = face_service.process_verification(doc)
    
    # 2. QR Code Decoding
    qr_res = decode_aadhaar_qr(doc.original_file.path)
    
    # 3. Risk Engine
    risk_res = risk_calc.evaluate_for_document(doc)
    
    print(f"1. Document face detected?   : {face_res['document_face_detected']}")
    print(f"2. Document face bbox?       : {face_res['document_face_bbox']}")
    print(f"3. Document face crop size?  : {face_res['document_face_crop_size']}")
    print(f"4. Selfie face detected?     : {face_res['selfie_face_detected']}")
    print(f"5. Liveness result?          : {face_res['liveness_status']}")
    print(f"6. Face similarity/distance? : Sim: {face_res['face_similarity']}, Dist: {face_res['face_distance']}")
    print(f"7. Threshold?                : {face_res['face_match_threshold']}")
    print(f"8. Face result?              : {face_res['final_face_status']}")
    print(f"9. QR result?                : {'DETECTED' if qr_res.get('qr_detected') else 'UNAVAILABLE'}")
    print(f"10. Final risk score?        : {risk_res['risk_score']}/100 ({risk_res['category']})")
    print(f"11. Decision narrative?      :\n  \"{face_res['explanation']}\"")

if __name__ == '__main__':
    prepare_test_assets()

    # Test A: Genuine Aadhaar Person A + Live selfie Person A (Same person)
    run_test_case(
        "TEST A: Aadhaar Person A + Live selfie Person A (Expected: MATCH)",
        "backend/media/test_suite/doc_genuine_aadhaar.jpg",
        "backend/media/test_suite/selfie_person_a.jpg"
    )

    # Test B: Genuine Aadhaar Person A + Live selfie Person B (Different person)
    run_test_case(
        "TEST B: Aadhaar Person A + Live selfie Person B (Expected: MISMATCH)",
        "backend/media/test_suite/doc_genuine_aadhaar.jpg",
        "backend/media/test_suite/selfie_person_b.jpg"
    )

    # Test C: Aadhaar + Poor / blurry selfie (Expected: INCONCLUSIVE / REVIEW REQUIRED)
    run_test_case(
        "TEST C: Aadhaar with portrait + Blurry selfie (Expected: INCONCLUSIVE / REVIEW)",
        "backend/media/test_suite/doc_genuine_aadhaar.jpg",
        "backend/media/test_suite/selfie_blurry.jpg"
    )

    # Test D: Aadhaar with no photo + Valid selfie (Expected: NOT AVAILABLE / REVIEW)
    run_test_case(
        "TEST D: Aadhaar with no usable portrait + Valid selfie (Expected: NOT AVAILABLE)",
        "backend/media/test_suite/doc_no_photo.jpg",
        "backend/media/test_suite/selfie_person_a.jpg"
    )
