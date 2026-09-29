import os
import sys
import django

sys.path.insert(0, os.path.abspath('backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from documents.models import Document, DocumentQualityAnalysis
from face_verification.models import FaceVerification
from tamper_detection.models import TamperAnalysis
from identity_verification.models import VerificationResult, IdentityConsistencyCheck
from ocr_engine.models import OCRAnalysis
from risk_engine.models import RiskFactor

print("=" * 80)
print("INSPECTING LATEST 5 DOCUMENTS IN DATABASE")
print("=" * 80)

docs = Document.objects.order_by('-created_at')[:5]
for d in docs:
    print(f"\n--- Document {d.verification_id} ---")
    print(f"Created: {d.created_at}")
    print(f"Type: {d.document_type}")
    print(f"File: {d.original_file}")
    print(f"Selfie: {d.selfie_file}")
    print(f"Status: {d.processing_status}")
    
    # Check verification result
    vr = getattr(d, 'verification_result', None)
    if vr:
        print(f"Risk Score: {vr.overall_risk_score}/100 ({vr.risk_category})")
        print(f"OCR Score: {vr.ocr_score}, Face Score: {vr.face_score}, Tamper Score: {vr.tamper_score}, Consistency: {vr.consistency_score}")
        print(f"Explanation: {vr.explanation}")
        factors = RiskFactor.objects.filter(verification_result=vr)
        for f in factors:
            print(f"  - Factor: {f.factor_name} | Val: {f.factor_value} | Contribution: +{f.contribution} | Desc: {f.description}")
    else:
        print("No VerificationResult found!")

    # Check Quality / Authenticity
    qa = getattr(d, 'quality_analysis', None)
    if qa:
        print(f"Quality Score: {qa.quality_score} | Classified: {qa.classified_type} (conf: {qa.classification_confidence})")
        print(f"Quality Metrics: {qa.metrics}")
    
    # Check Face
    fv = getattr(d, 'face_verification', None)
    if fv:
        print(f"Face: doc_face={fv.document_face_detected}, selfie_face={fv.selfie_face_detected}, sim={fv.similarity_score}, status={fv.match_status}, explanation={fv.explanation}")
    
    # Check Tamper
    ta = getattr(d, 'tamper_analysis', None)
    if ta:
        print(f"Tamper: prob={ta.tampering_probability}, risk={ta.risk_level}, signals={ta.signals}")
    
    # Check Consistency
    ic = getattr(d, 'consistency_check', None)
    if ic:
        print(f"Consistency: score={ic.consistency_score}, name={ic.name_match}, dob={ic.dob_match}")
