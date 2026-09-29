from django.urls import path
from .views import IdentityConsistencyResultsView, IdentityConsistencyAPIView, VerificationHistoryView

app_name = 'identity_verification'

urlpatterns = [
    path('history/', VerificationHistoryView.as_view(), name='history'),
    path('results/<str:verification_id>/', IdentityConsistencyResultsView.as_view(), name='results'),
    path('consistency/<str:verification_id>/', IdentityConsistencyAPIView.as_view(), name='consistency'),
]
