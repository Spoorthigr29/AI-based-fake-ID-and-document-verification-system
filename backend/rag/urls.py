from django.urls import path
from . import views

app_name = 'rag'

urlpatterns = [
    path('api/explain/', views.rag_explanation_api, name='api_explain'),
    path('api/rebuild-index/', views.rebuild_rag_index_api, name='api_rebuild_index'),
]
