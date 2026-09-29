from django.urls import path
from . import views

app_name = 'ocr_engine'

urlpatterns = [
    path('results/<str:verification_id>/', views.ocr_results_view, name='results'),
]
