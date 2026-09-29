from django.urls import path
from . import views

app_name = 'risk_engine'

urlpatterns = [
    path('status/', views.risk_engine_status_view, name='status'),
    path('ml/', views.MLRiskScoreAPIView.as_view(), name='ml_predict'),
]
