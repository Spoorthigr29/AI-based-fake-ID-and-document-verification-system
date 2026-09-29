from django.urls import path
from .views import TamperResultsView, TamperAnalysisAPIView

app_name = 'tamper_detection'

urlpatterns = [
    path('results/<str:verification_id>/', TamperResultsView.as_view(), name='results'),
    path('analyze/<str:verification_id>/', TamperAnalysisAPIView.as_view(), name='analyze'),
]
