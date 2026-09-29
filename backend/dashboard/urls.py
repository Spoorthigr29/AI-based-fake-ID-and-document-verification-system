from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('dashboard/', views.dashboard_view, name='main'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
]
