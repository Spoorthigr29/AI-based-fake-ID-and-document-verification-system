from django.urls import path
from .views import FaceVerificationResultsView, FaceVerificationAPIView

app_name = 'face_verification'

urlpatterns = [
    path('results/<str:verification_id>/', FaceVerificationResultsView.as_view(), name='results'),
    path('verify/<str:verification_id>/', FaceVerificationAPIView.as_view(), name='verify'),
]
