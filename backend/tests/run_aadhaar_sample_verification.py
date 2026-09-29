import os
import sys
import django

# Setup django
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.core.files import File
from documents.models import Document
from verification.services.verification_pipeline import VerificationPipelineOrchestrator

def run_tests():
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'media', 'test_suite'))
    orch = VerificationPipelineOrchestrator()

    # Test 1: Genuine Aadhaar + Matching Selfie
    print("\n================ TEST 1: Genuine Aadhaar + Live Selfie ================")
    doc_path = os.path.join(test_dir, 'doc_genuine_aadhaar.jpg')
    selfie_path = os.path.join(test_dir, 'selfie_person_a.jpg')

    if os.path.exists(doc_path) and os.path.exists(selfie_path):
        with open(doc_path, 'rb') as f_doc, open(selfie_path, 'rb') as f_selfie:
            doc = Document.objects.create(
                document_type='sample_aadhaar',
                original_file=File(f_doc, name='eval_genuine_aadhaar.jpg'),
                selfie_file=File(f_selfie, name='eval_selfie.jpg')
            )
            res = orch.run_pipeline(doc)
            print(f"Status: {res.get('status')}")
            print(f"Decision: {res.get('decision')}")
            print(f"Risk Score: {res.get('risk_score')} / 100")
            print(f"Category: {res.get('category')}")
            print(f"Explanation: {res.get('explanation')}")

    # Test 2: Blurry Document Image (Expect RESCAN)
    print("\n================ TEST 2: Blurry / Degraded Document Image ================")
    blurry_path = os.path.join(test_dir, 'doc_blurry.jpg')
    if os.path.exists(blurry_path):
        with open(blurry_path, 'rb') as f_doc:
            doc_blurry = Document.objects.create(
                document_type='sample_aadhaar',
                original_file=File(f_doc, name='eval_blurry_aadhaar.jpg')
            )
            res_b = orch.run_pipeline(doc_blurry)
            print(f"Status: {res_b.get('status')}")
            print(f"Decision: {res_b.get('decision')}")
            print(f"Risk Score: {res_b.get('risk_score')} / 100")
            print(f"Category: {res_b.get('category')}")
            print(f"Rescan Message: {res_b.get('rescan_message')}")

    # Test 3: Tampered Document Image (Expect Higher Risk / REJECT)
    print("\n================ TEST 3: Tampered Document Image ================")
    tamper_path = os.path.join(test_dir, 'doc_tampered_aadhaar.jpg')
    if os.path.exists(tamper_path):
        with open(tamper_path, 'rb') as f_doc:
            doc_tamper = Document.objects.create(
                document_type='sample_aadhaar',
                original_file=File(f_doc, name='eval_tampered_aadhaar.jpg')
            )
            res_t = orch.run_pipeline(doc_tamper)
            print(f"Status: {res_t.get('status')}")
            print(f"Decision: {res_t.get('decision')}")
            print(f"Risk Score: {res_t.get('risk_score')} / 100")
            print(f"Category: {res_t.get('category')}")
            print(f"Explanation: {res_t.get('explanation')}")

if __name__ == '__main__':
    run_tests()
