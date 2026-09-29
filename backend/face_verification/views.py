from django.shortcuts import render, get_object_or_404
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from documents.models import Document
from face_verification.models import FaceVerification
from face_verification.services.face_service import FaceVerificationService

class FaceVerificationAPIView(APIView):
    """
    POST /api/face-verification/<verification_id>/
    Trigger biometric face detection, crop extraction, and feature comparison between
    identity document and selfie photo.
    """
    def post(self, request, verification_id):
        document = get_object_or_404(Document, verification_id=verification_id)

        try:
            service = FaceVerificationService()
            result = service.process_verification(document)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "document_face_detected": False,
                "selfie_face_detected": False,
                "similarity_score": 0.0,
                "confidence": 0.0,
                "match_status": "MANUAL_REVIEW",
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request, verification_id):
        """Get current face verification status and scores."""
        document = get_object_or_404(Document, verification_id=verification_id)
        face_result = getattr(document, 'face_verification', None)

        if not face_result:
            return Response({
                "document_face_detected": False,
                "selfie_face_detected": False,
                "similarity_score": 0.0,
                "confidence": 0.0,
                "match_status": "MANUAL_REVIEW",
                "message": "Face verification not yet executed for this document."
            }, status=status.HTTP_200_OK)

        return Response({
            "document_face_detected": face_result.document_face_detected,
            "selfie_face_detected": face_result.selfie_face_detected,
            "similarity_score": face_result.similarity_score,
            "confidence": face_result.confidence,
            "match_status": face_result.match_status,
            "issues": face_result.issues,
            "face_quality_metrics": face_result.face_quality_metrics,
        }, status=status.HTTP_200_OK)

class FaceVerificationResultsView(View):
    """HTML view for inspecting biometric face comparison results."""
    def get(self, request, verification_id):
        document = get_object_or_404(Document, verification_id=verification_id)
        face_result = getattr(document, 'face_verification', None)
        
        context = {
            'document': document,
            'face_result': face_result,
        }
        return render(request, 'face_verification/results.html', context)
