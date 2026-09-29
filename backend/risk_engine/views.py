from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.shortcuts import get_object_or_404

from documents.models import Document
from .services.risk_calculator import DeterministicRiskCalculator
from .services.ml_risk_predictor import MLRiskPredictor


class RiskScoreCalculationAPIView(APIView):
    """
    POST /api/risk-score/<verification_id>/
    Deterministic and explainable risk calculation endpoint for VerifyX AI.
    Consumes signals from Document analysis or explicit JSON payload.
    """

    def post(self, request, verification_id):
        calculator = DeterministicRiskCalculator()
        custom_data = request.data if isinstance(request.data, dict) else {}

        # 1. Try to find Document in DB
        doc = Document.objects.filter(verification_id=verification_id).first()
        if not doc and verification_id.isdigit():
            doc = Document.objects.filter(pk=int(verification_id)).first()

        try:
            if doc:
                # If custom data is provided, calculate directly or evaluate DB document
                if custom_data:
                    # Merge DB signals with custom overrides
                    result = calculator.calculate_risk(custom_data)
                    result["verification_id"] = doc.verification_id
                else:
                    result = calculator.evaluate_for_document(doc)
            else:
                # If document is not in DB, evaluate using custom_data payload
                if not custom_data:
                    return Response({
                        "error": f"Document '{verification_id}' not found and no screening features provided in request body."
                    }, status=status.HTTP_404_NOT_FOUND)
                
                result = calculator.calculate_risk(custom_data)
                result["verification_id"] = verification_id

            return Response({
                "risk_score": result["risk_score"],
                "category": result["category"],
                "factors": result["factors"],
                "explanation": result["explanation"]
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                "error": f"Risk calculation failed: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request, verification_id):
        """Fetch latest calculated risk result for a document."""
        calculator = DeterministicRiskCalculator()
        doc = get_object_or_404(Document, verification_id=verification_id)
        try:
            result = calculator.evaluate_for_document(doc)
            return Response({
                "risk_score": result["risk_score"],
                "category": result["category"],
                "factors": result["factors"],
                "explanation": result["explanation"]
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "error": f"Failed to retrieve risk score: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MLRiskScoreAPIView(APIView):
    """
    POST /api/risk-score/ml/
    Inference endpoint for ML-based risk category classification using trained model pipeline.
    """
    def post(self, request):
        data = request.data
        if not isinstance(data, dict):
            return Response({
                "success": False,
                "error": "Request body must be a valid JSON object with screening features."
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            predictor = MLRiskPredictor()
            result = predictor.predict_risk(data)

            response_payload = {
                "success": True,
                "risk_category": result.risk_category,
                "confidence": result.confidence,
                "model": result.model_name,
                "probabilities": result.probabilities
            }
            if result.warning:
                response_payload["warning"] = result.warning

            return Response(response_payload, status=status.HTTP_200_OK)

        except ValueError as ve:
            return Response({
                "success": False,
                "error": str(ve)
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({
                "success": False,
                "error": f"Inference error: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        """Metadata / status of the ML Risk Predictor."""
        predictor = MLRiskPredictor()
        return Response({
            "status": "ready" if predictor.pipeline is not None else "fallback_mode",
            "model_name": predictor.metadata.get('model_name', 'Not Loaded'),
            "features_required": predictor.feature_names,
            "classes": predictor.classes,
            "training_date": predictor.metadata.get('training_date'),
            "disclaimer": predictor.metadata.get('disclaimer')
        }, status=status.HTTP_200_OK)


def risk_engine_status_view(request):
    """Status view for risk engine health."""
    predictor = MLRiskPredictor()
    return JsonResponse({
        'status': 'ready',
        'engine': 'risk_engine',
        'ml_model_loaded': predictor.pipeline is not None,
        'model_name': predictor.metadata.get('model_name', 'Rule-Based Heuristics')
    })
