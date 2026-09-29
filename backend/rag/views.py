import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status

from .services.rag_service import RAGService

@api_view(['POST'])
@permission_classes([AllowAny])
def rag_explanation_api(request):
    """
    Internal RAG Explainability API Endpoint.
    Accepts:
        {
            "document_type": "AADHAAR" | "PAN" | "PASSPORT" | "DRIVING_LICENSE" | "VOTER_ID",
            "extracted_text": "...",
            "existing_analysis_results": { ... }
        }
    Returns:
        {
            "document_type": "...",
            "evidence_found": [...],
            "relevant_rules": [...],
            "retrieved_context": [...],
            "risk_assessment": { ... },
            "explanation": "..."
        }
    """
    try:
        data = request.data if hasattr(request, 'data') else json.loads(request.body)
        document_type = data.get('document_type', 'OTHER')
        extracted_text = data.get('extracted_text', '')
        analysis_results = data.get('existing_analysis_results', {})

        rag_service = RAGService.get_instance()
        explanation_payload = rag_service.generate_explanation(
            document_type=document_type,
            extracted_text=extracted_text,
            analysis_results=analysis_results
        )

        return Response(explanation_payload, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            {
                "error": f"RAG execution error: {str(e)}",
                "explanation": "Standard verification assessment completed. Knowledge retrieval was temporarily unavailable."
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([AllowAny])
def rebuild_rag_index_api(request):
    """Admin/System API to rebuild the local vector store index."""
    try:
        rag_service = RAGService.get_instance()
        num_chunks = rag_service.rebuild_index()
        return Response({
            "status": "SUCCESS",
            "message": f"RAG vector index successfully rebuilt with {num_chunks} chunks."
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            "status": "ERROR",
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
