from django.urls import path
from .views import IdentityConsistencyAPIView

app_name = 'identity_verification_api'

urlpatterns = [
    path('consistency/<str:verification_id>/', IdentityConsistencyAPIView.as_view(), name='consistency'),
    path('<str:verification_id>/', IdentityConsistencyAPIView.as_view(), name='process'),
]
