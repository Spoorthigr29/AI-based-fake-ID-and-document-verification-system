from django.urls import path
from .views import RiskScoreCalculationAPIView, MLRiskScoreAPIView

app_name = 'risk_engine_api'

urlpatterns = [
    path('ml/', MLRiskScoreAPIView.as_view(), name='ml_predict'),
    path('predict/', MLRiskScoreAPIView.as_view(), name='predict'),
    path('<str:verification_id>/', RiskScoreCalculationAPIView.as_view(), name='calculate_risk'),
]
