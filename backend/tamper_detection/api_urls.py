from django.urls import path
from .views import TamperAnalysisAPIView

app_name = 'tamper_detection_api'

urlpatterns = [
    path('analyze/<str:verification_id>/', TamperAnalysisAPIView.as_view(), name='analyze'),
    path('<str:verification_id>/', TamperAnalysisAPIView.as_view(), name='process'),
]
