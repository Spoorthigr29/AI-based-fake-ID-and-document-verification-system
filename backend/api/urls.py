"""
VerifyX AI — Unified API URL Routing
====================================
Routes API traffic to respective subsystem controllers:
- /api/verification/ & /api/documents/ -> Document Management & Capabilities
- /api/ocr/ -> OCR Text Extraction & Layout Analysis
- /api/face-verification/ -> Biometric Face Detection & Verification
- /api/tamper/ -> Forensic Tampering & ELA Analysis
- /api/verify/ & /api/identity-verification/ -> Cross-Field Consistency Checks
- /api/risk-score/ & /api/risk/ -> Risk Calculation & ML Scoring
"""

from django.urls import path, include

app_name = 'api'

urlpatterns = [
    path('verification/', include(('documents.api_urls', 'verification_api'), namespace='verification_api')),
    path('documents/', include(('documents.api_urls', 'documents_api'), namespace='documents_api')),
    path('ocr/', include(('ocr_engine.api_urls', 'ocr_api'), namespace='ocr_api')),
    path('face-verification/', include(('face_verification.api_urls', 'face_api'), namespace='face_api')),
    path('tamper/', include(('tamper_detection.api_urls', 'tamper_api'), namespace='tamper_api')),
    path('verify/', include(('identity_verification.api_urls', 'verify_api'), namespace='verify_api')),
    path('identity-verification/', include(('identity_verification.api_urls', 'identity_api'), namespace='identity_api')),
    path('risk-score/', include(('risk_engine.api_urls', 'risk_score_api'), namespace='risk_score_api')),
    path('risk/', include(('risk_engine.api_urls', 'risk_api'), namespace='risk_api')),
]
