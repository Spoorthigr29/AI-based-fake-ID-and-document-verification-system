import sys, os, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings'
import django; django.setup()
from face_verification.services.face_detector import FaceDetector

fd = FaceDetector()
img = cv2.imread('backend/media/test_suite/doc_genuine_aadhaar.jpg')
cands = fd._find_candidate_faces(img, is_document=True)
print(f'Found {len(cands)} candidates on doc_genuine_aadhaar:')
for idx, c in enumerate(cands):
    print(f"  Candidate {idx}: box={c['box']}, conf={c['confidence']}, zone={c['zone']}")

img_np = cv2.imread('backend/media/test_suite/doc_no_photo.jpg')
cands_np = fd._find_candidate_faces(img_np, is_document=True)
print(f'Found {len(cands_np)} candidates on doc_no_photo:')
for idx, c in enumerate(cands_np):
    print(f"  Candidate {idx}: box={c['box']}, conf={c['confidence']}, zone={c['zone']}")
