from django.shortcuts import render, get_object_or_404
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from documents.models import Document
from identity_verification.models import IdentityConsistencyCheck
from identity_verification.services.consistency_engine import IdentityConsistencyEngine

class IdentityConsistencyAPIView(APIView):
    """
    POST /api/verify/consistency/<verification_id>/
    Trigger cross-source and cross-field identity consistency verification.
    """
    def post(self, request, verification_id):
        document = get_object_or_404(Document, verification_id=verification_id)
        claims = request.data.get("claims") if isinstance(request.data, dict) else None

        try:
            engine = IdentityConsistencyEngine()
            result = engine.process_consistency(document, applicant_claims=claims)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "name_match": False,
                "dob_match": False,
                "document_number_match": False,
                "address_match": False,
                "consistency_score": 0,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request, verification_id):
        """Retrieve stored identity consistency check."""
        document = get_object_or_404(Document, verification_id=verification_id)
        check = getattr(document, 'consistency_check', None)

        if not check:
            return Response({
                "name_match": False,
                "dob_match": False,
                "document_number_match": False,
                "address_match": False,
                "consistency_score": 0,
                "message": "Consistency check not yet executed."
            }, status=status.HTTP_200_OK)

        return Response({
            "name_match": check.name_match,
            "dob_match": check.dob_match,
            "document_number_match": check.document_number_match,
            "address_match": check.address_match,
            "consistency_score": check.consistency_score,
            "status": check.status,
            "checklist": check.checklist,
            "field_details": check.field_details
        }, status=status.HTTP_200_OK)

class IdentityConsistencyResultsView(View):
    """HTML View for inspecting field-by-field consistency breakdown."""
    def get(self, request, verification_id):
        document = get_object_or_404(Document, verification_id=verification_id)
        consistency_check = getattr(document, 'consistency_check', None)

        context = {
            'document': document,
            'consistency_check': consistency_check,
        }
        return render(request, 'identity_verification/results.html', context)

class VerificationHistoryView(View):
    """HTML View for viewing history of processed identity verifications."""
    def get(self, request):
        documents = Document.objects.all().order_by('-created_at')[:50]
        return render(request, 'identity_verification/history.html', {'documents': documents})
