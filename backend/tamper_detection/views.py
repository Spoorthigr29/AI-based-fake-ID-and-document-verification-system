from django.shortcuts import render, get_object_or_404
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from documents.models import Document
from tamper_detection.models import TamperAnalysis
from tamper_detection.services.tamper_analyzer import TamperAnalyzerService

class TamperAnalysisAPIView(APIView):
    """
    POST /api/tamper/analyze/<verification_id>/
    Trigger forensic digital tampering inspection (ELA, artifacts, boundaries, copy-move).
    """
    def post(self, request, verification_id):
        document = get_object_or_404(Document, verification_id=verification_id)

        try:
            service = TamperAnalyzerService()
            result = service.process_document(document)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "tampering_probability": 0.0,
                "risk_level": "LOW",
                "signals": [f"Error processing tampering detection: {str(e)}"],
                "suspicious_regions": []
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request, verification_id):
        """Retrieve stored tampering analysis result."""
        document = get_object_or_404(Document, verification_id=verification_id)
        tamper_res = getattr(document, 'tamper_analysis', None)

        if not tamper_res:
            return Response({
                "tampering_probability": 0.0,
                "risk_level": "LOW",
                "signals": ["Tampering analysis has not been executed yet."],
                "suspicious_regions": []
            }, status=status.HTTP_200_OK)

        return Response({
            "tampering_probability": tamper_res.tampering_probability,
            "risk_level": tamper_res.risk_level,
            "signals": tamper_res.signals,
            "suspicious_regions": tamper_res.suspicious_regions,
            "annotated_image": tamper_res.annotated_image,
            "ela_image": tamper_res.ela_image
        }, status=status.HTTP_200_OK)

class TamperResultsView(View):
    """HTML Forensic Dashboard for inspecting tampering analysis, ELA, and heatmaps."""
    def get(self, request, verification_id):
        document = get_object_or_404(Document, verification_id=verification_id)
        tamper_analysis = getattr(document, 'tamper_analysis', None)

        context = {
            'document': document,
            'tamper_analysis': tamper_analysis,
        }
        return render(request, 'tamper_detection/results.html', context)
