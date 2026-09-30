"""
Django settings for VERIFYX AI project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent

# Dataset & Model Artifact Directories
DATASET_DIR = BASE_DIR / 'dataset'
DATASET_PATH = DATASET_DIR / 'verifyx_120_records_training_dataset.csv'
ML_MODELS_DIR = BASE_DIR / 'ml_models'

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-verifyx-ai-hackathon-prototype-key-998822')

DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes')

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,testserver').split(',')


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party Apps
    'rest_framework',
    'corsheaders',

    # VERIFYX Custom Modular Apps
    'accounts.apps.AccountsConfig',
    'documents.apps.DocumentsConfig',
    'ocr_engine.apps.OcrEngineConfig',
    'face_verification.apps.FaceVerificationConfig',
    'tamper_detection.apps.TamperDetectionConfig',
    'identity_verification.apps.IdentityVerificationConfig',
    'risk_engine.apps.RiskEngineConfig',
    'reports.apps.ReportsConfig',
    'dashboard.apps.DashboardConfig',
    'audit.apps.AuditConfig',
    'rag.apps.RagConfig',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            ROOT_DIR / 'frontend' / 'templates',
            BASE_DIR / 'templates',
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'
ASGI_APPLICATION = 'config.asgi.application'


# Database
# Default is SQLite for development, Postgres compatible
DATABASES = {
    'default': {
        'ENGINE': os.getenv('DB_ENGINE', 'django.db.backends.sqlite3'),
        'NAME': BASE_DIR / 'db.sqlite3' if os.getenv('DB_ENGINE') is None else os.getenv('DB_NAME', 'verifyx_db'),
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    ROOT_DIR / 'frontend' / 'static',
]
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files (Uploaded identity documents)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# Authentication URLs
LOGIN_URL = 'accounts:login'
LOGIN_REDIRECT_URL = 'dashboard:home'
LOGOUT_REDIRECT_URL = 'accounts:login'

# Security & Session Hardening
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = False
SESSION_COOKIE_AGE = 86400  # 24 hours
X_FRAME_OPTIONS = 'DENY'
SECURE_CONTENT_TYPE_NOSNIFF = True

# File Upload Security
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB Max
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB Max
ALLOWED_DOCUMENT_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.pdf']

# Privacy & Data Retention Configuration
DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '90'))
AUTO_PURGE_EXPIRED_DATA = os.getenv('AUTO_PURGE_EXPIRED_DATA', 'False').lower() in ('true', '1', 'yes')
MASK_PII_IN_AUDIT_LOGS = os.getenv('MASK_PII_IN_AUDIT_LOGS', 'True').lower() in ('true', '1', 'yes')

# VERIFYX System Thresholds
AI_MODEL_CONFIDENCE_THRESHOLD = float(os.getenv('AI_MODEL_CONFIDENCE_THRESHOLD', '0.85'))
MANUAL_REVIEW_RISK_THRESHOLD = float(os.getenv('MANUAL_REVIEW_RISK_THRESHOLD', '60.0'))
FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD', '0.70'))
FACE_MANUAL_REVIEW_THRESHOLD = float(os.getenv('FACE_MANUAL_REVIEW_THRESHOLD', '0.50'))
FACE_MIN_QUALITY_SCORE = float(os.getenv('FACE_MIN_QUALITY_SCORE', '40.0'))

# Cross-Origin Resource Sharing (CORS) for Netlify & External Frontends
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

