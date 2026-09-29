"""
URL configuration for VERIFYX AI project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from accounts import views as account_views
from dashboard import views as dashboard_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('dashboard.urls', namespace='dashboard')),
    path('login/', account_views.login_view, name='login_direct'),
    path('logout/', account_views.logout_view, name='logout_direct'),
    path('analytics/', dashboard_views.analytics_view, name='analytics_direct'),
    path('admin-dashboard/', dashboard_views.admin_dashboard_view, name='admin_dashboard_direct'),
    path('verification/', include(('documents.urls', 'verification'), namespace='verification')),
    path('documents/', include(('documents.urls', 'documents'), namespace='documents')),
    path('api/', include(('api.urls', 'api'), namespace='api')),
    path('api/verification/', include(('documents.api_urls', 'verification_api'), namespace='verification_api')),
    path('api/documents/', include(('documents.api_urls', 'documents_api'), namespace='documents_api')),
    path('api/ocr/', include(('ocr_engine.api_urls', 'ocr_api'), namespace='ocr_api')),
    path('api/face-verification/', include(('face_verification.api_urls', 'face_api'), namespace='face_api')),
    path('api/tamper/', include(('tamper_detection.api_urls', 'tamper_api'), namespace='tamper_api')),
    path('api/verify/', include(('identity_verification.api_urls', 'verify_api'), namespace='verify_api')),
    path('api/identity-verification/', include(('identity_verification.api_urls', 'identity_api'), namespace='identity_api')),
    path('api/risk-score/', include(('risk_engine.api_urls', 'risk_score_api'), namespace='risk_score_api')),
    path('api/risk/', include(('risk_engine.api_urls', 'risk_api'), namespace='risk_api')),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('ocr/', include('ocr_engine.urls', namespace='ocr_engine')),
    path('face/', include('face_verification.urls', namespace='face_verification')),
    path('tamper/', include('tamper_detection.urls', namespace='tamper_detection')),
    path('verify/', include('identity_verification.urls', namespace='identity_verification')),
    path('risk/', include('risk_engine.urls', namespace='risk_engine')),
    path('reports/', include('reports.urls', namespace='reports')),
    path('audit/', include('audit.urls', namespace='audit')),
    path('rag/', include(('rag.urls', 'rag'), namespace='rag')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
