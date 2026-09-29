from django.urls import path
from .api_views import (
    DocumentUploadAPIView,
    DocumentAnalyzeAPIView,
    StartVerificationPipelineAPIView,
    VerificationDebugAPIView
)

app_name = 'documents_api'

urlpatterns = [
    path('upload/', DocumentUploadAPIView.as_view(), name='api_upload'),
    path('analyze/<str:verification_id>/', DocumentAnalyzeAPIView.as_view(), name='api_analyze'),
    path('start/<str:verification_id>/', StartVerificationPipelineAPIView.as_view(), name='api_start'),
    path('debug/<str:verification_id>/', VerificationDebugAPIView.as_view(), name='api_debug'),
    path('<str:verification_id>/debug/', VerificationDebugAPIView.as_view(), name='api_debug_alias'),
]

