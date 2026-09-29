from django.urls import path
from .api_views import OCRProcessAPIView

app_name = 'ocr_api'

urlpatterns = [
    path('process/<str:verification_id>/', OCRProcessAPIView.as_view(), name='api_process'),
]
