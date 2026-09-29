import os
import sys
import django
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from documents.models import Document
from face_verification.services.face_detector import FaceDetector
from face_verification.services.face_service import FaceVerificationService
from ml.predict_document import DocumentAuthenticityPredictor

doc = Document.objects.order_by('-created_at').first()
print("Inspecting document:", doc.verification_id)
print("File:", doc.original_file.path)
if doc.selfie_file:
    print("Selfie:", doc.selfie_file.path)

# 1. Test Authenticity Predictor
predictor = DocumentAuthenticityPredictor()
auth_res = predictor.predict(doc.original_file.path)
print("\n--- Authenticity Prediction ---")
print(auth_res)

# 2. Test Face Detector
detector = FaceDetector()
doc_res = detector.detect_faces(doc.original_file.path, is_document=True)
print("\n--- Document Face Detection ---")
print(f"Face count: {doc_res.face_count}, status: {doc_res.status}, issues: {doc_res.issues}")

if doc.selfie_file:
    selfie_res = detector.detect_faces(doc.selfie_file.path, is_document=False)
    print("\n--- Selfie Face Detection ---")
    print(f"Face count: {selfie_res.face_count}, status: {selfie_res.status}")

# 3. Test Full Face Service
face_svc = FaceVerificationService()
face_res = face_svc.process_verification(doc)
print("\n--- Face Verification Result ---")
print(face_res)
