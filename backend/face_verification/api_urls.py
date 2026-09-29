from django.urls import path
from .views import FaceVerificationAPIView

app_name = 'face_verification_api'

urlpatterns = [
    path('<str:verification_id>/', FaceVerificationAPIView.as_view(), name='process'),
]
