"""
VerifyX AI - Automated Critical Test Cases Evaluation Suite
===========================================================
Executes all 10 standard evaluation scenarios specified in the system specification:
TEST 1: Genuine document + matching live person
TEST 2: Genuine document + different person's selfie
TEST 3: Tampered document + matching live person
TEST 4: Tampered document + different person's selfie
TEST 5: Wrong document type
TEST 6: Blurry document
TEST 7: Blurry selfie
TEST 8: Document with no usable photograph
TEST 9: Multiple faces in selfie
TEST 10: Unreadable document
"""

import os
import sys
import django
import cv2
import numpy as np

# Setup Django Environment
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from documents.models import Document
from verification.services.verification_pipeline import VerificationPipelineOrchestrator
from documents.services.document_classifier import DocumentClassifier
from documents.services.quality_analyzer import DocumentQualityAnalyzer
from face_verification.services.face_service import FaceVerificationService
from tamper_detection.services.tamper_analyzer import TamperAnalyzerService
from identity_verification.services.consistency_engine import IdentityConsistencyEngine
from risk_engine.services.risk_calculator import DeterministicRiskCalculator

from django.conf import settings

def create_synthetic_test_assets():
    """Generates synthetic test images for each evaluation scenario using realistic face crops."""
    test_suite_dir = os.path.join(settings.MEDIA_ROOT, 'test_suite')
    os.makedirs(test_suite_dir, exist_ok=True)
    
    # Real applicant face crop
    sample_selfie_path = os.path.join(settings.MEDIA_ROOT, 'selfies', '2026', '09', '27', 'f74719ef8caa47358c3b82fef91d75fb.jpg')
    if os.path.exists(sample_selfie_path):
        face_a = cv2.imread(sample_selfie_path)
    else:
        face_a = np.ones((400, 300, 3), dtype=np.uint8) * 200
        
    cv2.imwrite(os.path.join(test_suite_dir, 'selfie_person_a.jpg'), face_a)

    # Base Face 2 (Applicant B - Distinct Synthetic Person with different geometry)
    face_b = np.ones((400, 300, 3), dtype=np.uint8) * 180
    # Add head silhouette
    cv2.ellipse(face_b, (150, 200), (90, 120), 0, 0, 360, (210, 190, 170), -1)
    # Hair
    cv2.ellipse(face_b, (150, 120), (95, 60), 0, 180, 360, (30, 20, 15), -1)
    # Eyes
    cv2.circle(face_b, (115, 175), 12, (255, 255, 255), -1)
    cv2.circle(face_b, (115, 175), 6, (40, 25, 15), -1)
    cv2.circle(face_b, (185, 175), 12, (255, 255, 255), -1)
    cv2.circle(face_b, (185, 175), 6, (40, 25, 15), -1)
    # Nose & Mouth
    cv2.line(face_b, (150, 180), (150, 215), (160, 140, 120), 3)
    cv2.ellipse(face_b, (150, 245), (35, 12), 0, 0, 180, (140, 70, 70), -1)
    cv2.imwrite(os.path.join(test_suite_dir, 'selfie_person_b.jpg'), face_b)

    # Multi-face selfie
    fa_small = cv2.resize(face_a, (250, 350))
    fb_small = cv2.resize(face_b, (250, 350))
    multi_face = np.hstack([fa_small, fb_small])
    cv2.imwrite(os.path.join(test_suite_dir, 'selfie_multi_face.jpg'), multi_face)

    # Blurry selfie
    blurry_selfie = cv2.GaussianBlur(face_a, (45, 45), 0)
    cv2.imwrite(os.path.join(test_suite_dir, 'selfie_blurry.jpg'), blurry_selfie)

    # Genuine Aadhaar Document with Face A
    real_doc_path = os.path.join(settings.MEDIA_ROOT, 'documents', '2026', '09', '27', 'c77a3e27e14d4eeebebd6871709f2808.jpg')
    if os.path.exists(real_doc_path):
        doc_aadhaar = cv2.imread(real_doc_path)
    else:
        doc_aadhaar = np.ones((600, 900, 3), dtype=np.uint8) * 245
    cv2.imwrite(os.path.join(test_suite_dir, 'doc_genuine_aadhaar.jpg'), doc_aadhaar)

    # Tampered Aadhaar Document (Spliced number & copy-move texture)
    doc_tampered = doc_aadhaar.copy()
    cv2.rectangle(doc_tampered, (290, 440), (620, 510), (255, 255, 255), -1)
    cv2.putText(doc_tampered, "9999 0000 1111", (300, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
    cv2.imwrite(os.path.join(test_suite_dir, 'doc_tampered_aadhaar.jpg'), doc_tampered)

    # PAN Card Document
    doc_pan = np.ones((600, 900, 3), dtype=np.uint8) * 235
    cv2.putText(doc_pan, "INCOME TAX DEPARTMENT", (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 0, 0), 2)
    cv2.putText(doc_pan, "GOVT. OF INDIA", (340, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    doc_face_resized = cv2.resize(face_a, (160, 200))
    doc_pan[140:340, 50:210] = doc_face_resized
    cv2.putText(doc_pan, "Name: SPOORTHI G R", (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(doc_pan, "Father's Name: RAJESH G", (240, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(doc_pan, "DOB: 29/03/2006", (240, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(doc_pan, "ABCDE1234F", (240, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    cv2.imwrite(os.path.join(test_suite_dir, 'doc_genuine_pan.jpg'), doc_pan)

    # Blurry Document
    blurry_doc = cv2.GaussianBlur(doc_aadhaar, (45, 45), 0)
    cv2.imwrite(os.path.join(test_suite_dir, 'doc_blurry.jpg'), blurry_doc)

    # Document with No Photo
    doc_no_photo = doc_aadhaar.copy()
    doc_no_photo[140:340, 50:210] = 245
    cv2.imwrite(os.path.join(test_suite_dir, 'doc_no_photo.jpg'), doc_no_photo)

    # Unreadable Document
    unreadable_doc = np.random.randint(0, 256, (600, 900, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_suite_dir, 'doc_unreadable.jpg'), unreadable_doc)

def run_all_tests():
    create_synthetic_test_assets()
    orchestrator = VerificationPipelineOrchestrator()

    test_matrix = [
        {
            "test_id": "TEST 1",
            "name": "Genuine Document + Matching Live Person",
            "doc_path": "test_suite/doc_genuine_aadhaar.jpg",
            "selfie_path": "test_suite/selfie_person_a.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "LOW"
        },
        {
            "test_id": "TEST 2",
            "name": "Genuine Document + Different Person's Selfie",
            "doc_path": "test_suite/doc_genuine_aadhaar.jpg",
            "selfie_path": "test_suite/selfie_person_b.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "HIGH / MANUAL_REVIEW"
        },
        {
            "test_id": "TEST 3",
            "name": "Tampered Document + Matching Live Person",
            "doc_path": "test_suite/doc_tampered_aadhaar.jpg",
            "selfie_path": "test_suite/selfie_person_a.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "HIGH / MANUAL_REVIEW"
        },
        {
            "test_id": "TEST 4",
            "name": "Tampered Document + Different Person's Selfie",
            "doc_path": "test_suite/doc_tampered_aadhaar.jpg",
            "selfie_path": "test_suite/selfie_person_b.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "HIGH_RISK"
        },
        {
            "test_id": "TEST 5",
            "name": "Wrong Document Type (Selected Aadhaar, Uploaded PAN)",
            "doc_path": "test_suite/doc_genuine_pan.jpg",
            "selfie_path": "test_suite/selfie_person_a.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "MISMATCH_DETECTED"
        },
        {
            "test_id": "TEST 6",
            "name": "Blurry Document",
            "doc_path": "test_suite/doc_blurry.jpg",
            "selfie_path": "test_suite/selfie_person_a.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "QUALITY_WARNING"
        },
        {
            "test_id": "TEST 7",
            "name": "Blurry Selfie",
            "doc_path": "test_suite/doc_genuine_aadhaar.jpg",
            "selfie_path": "test_suite/selfie_blurry.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "QUALITY_WARNING"
        },
        {
            "test_id": "TEST 8",
            "name": "Document with No Usable Photograph",
            "doc_path": "test_suite/doc_no_photo.jpg",
            "selfie_path": "test_suite/selfie_person_a.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "FACE_NOT_APPLICABLE / REVIEW"
        },
        {
            "test_id": "TEST 9",
            "name": "Multiple Faces in Selfie",
            "doc_path": "test_suite/doc_genuine_aadhaar.jpg",
            "selfie_path": "test_suite/selfie_multi_face.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "MULTI_FACE_WARNING"
        },
        {
            "test_id": "TEST 10",
            "name": "Unreadable Document",
            "doc_path": "test_suite/doc_unreadable.jpg",
            "selfie_path": "test_suite/selfie_person_a.jpg",
            "doc_type": "AADHAAR",
            "expected_risk": "UNREADABLE_QUALITY"
        }
    ]

    print("\n" + "=" * 90)
    print("VERIFYX AI - 10 CRITICAL FORENSIC TEST CASES EVALUATION REPORT")
    print("=" * 90 + "\n")

    results_table = []

    for t in test_matrix:
        doc = Document.objects.create(
            document_type=t["doc_type"],
            original_file=t["doc_path"],
            selfie_file=t["selfie_path"]
        )

        res = orchestrator.run_pipeline(doc)
        
        # Harvest layer metrics
        doc.refresh_from_db()
        q_score = doc.quality_analysis.quality_score if hasattr(doc, 'quality_analysis') else 'N/A'
        ocr_conf = int(doc.ocr_analysis.overall_confidence * 100) if hasattr(doc, 'ocr_analysis') else 'N/A'
        face_match = doc.face_verification.match_status if hasattr(doc, 'face_verification') else 'N/A'
        face_sim = f"{int(doc.face_verification.similarity_score * 100)}%" if (hasattr(doc, 'face_verification') and doc.face_verification.similarity_score is not None) else 'N/A'
        tamper_prob = f"{int(doc.tamper_analysis.tampering_probability * 100)}%" if hasattr(doc, 'tamper_analysis') else 'N/A'
        consistency = f"{doc.consistency_check.consistency_score}%" if hasattr(doc, 'consistency_check') else 'N/A'
        
        row = {
            "Test ID": t["test_id"],
            "Scenario": t["name"],
            "Detected Type": doc.quality_analysis.classified_type if hasattr(doc, 'quality_analysis') else 'N/A',
            "Quality": q_score,
            "OCR Conf": f"{ocr_conf}%",
            "Tamper": tamper_prob,
            "Doc Face": "YES" if (hasattr(doc, 'face_verification') and doc.face_verification.document_face_detected) else "NO",
            "Selfie Face": "YES" if (hasattr(doc, 'face_verification') and doc.face_verification.selfie_face_detected) else "NO",
            "Face Match": f"{face_match} ({face_sim})",
            "Consistency": consistency,
            "Risk Score": f"{res['risk_score']}/100",
            "Risk Level": res['category'],
            "Status": res['status'],
            "Reasons": res['explanation'][:100] + "..."
        }
        results_table.append(row)

        print(f"[{t['test_id']}] {t['name']}")
        print(f"  -> Risk Score: {res['risk_score']}/100 | Level: {res['category']} | Decision: {res['status']}")
        print(f"  -> Quality: {q_score} | OCR: {ocr_conf}% | Tamper: {tamper_prob} | Face Match: {face_match} ({face_sim})")
        print(f"  -> Explanation: {res['explanation']}\n")
        sys.stdout.flush()

    return results_table

if __name__ == '__main__':
    run_all_tests()
