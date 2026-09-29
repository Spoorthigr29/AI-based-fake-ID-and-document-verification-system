from django.urls import path
from . import views

app_name = 'documents'

urlpatterns = [
    path('', views.verification_list_view, name='list'),
    path('history/', views.verification_list_view, name='history'),
    path('upload/', views.verification_upload_view, name='upload'),
    path('new/', views.verification_upload_view, name='new'),
    path('file/<str:verification_id>/<str:file_type>/', views.secure_document_file_view, name='secure_file'),
    path('demo/<str:case_name>/', views.demo_launch_view, name='demo_launch'),
    path('review-note/<str:verification_id>/', views.add_review_note_view, name='add_review_note'),
    path('processing/<str:verification_id>/', views.verification_processing_view, name='processing'),
    path('result/<str:verification_id>/', views.verification_detail_view, name='result'),
    path('<str:verification_id>/', views.verification_detail_view, name='detail'),
]
