"""
=============================================================
  DJANGO SETTINGS
=============================================================
Configured for:
  - Local development (DEBUG=True)
  - Google Cloud Run deployment (via environment variables)

CHANGE vs original:
  - Added python-dotenv load at the very top so values in your
    .env file are picked up by all the os.environ.get() calls below.
    Every other line is identical to the original.
=============================================================
"""

import os
from pathlib import Path

# Load .env file — must be the very first thing before any os.environ.get().
# If python-dotenv is not installed this silently does nothing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Base directory (project root) ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security ──────────────────────────────────────────────────────────────────
# IMPORTANT: Set SECRET_KEY as an environment variable in production!
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-key-change-this-in-production-!@#$%"
)

# Set DEBUG=False in production (Cloud Run sets this via env var)
DEBUG = os.environ.get("DEBUG", "True") == "True"

# Allowed hosts — Cloud Run provides its own domain
ALLOWED_HOSTS = os.environ.get(
    "ALLOWED_HOSTS",
    "localhost,127.0.0.1,0.0.0.0"
).split(",")

# Also allow any Cloud Run URL (*.run.app)
ALLOWED_HOSTS += ["*.run.app"]

# ── Application definition ────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.contenttypes",   # required by auth
    "django.contrib.sessions",        # for conversation history storage
    "django.contrib.staticfiles",     # for serving static files
    "chat",                           # our chat application
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # serves static files efficiently
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "myproject.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,   # looks for templates in <app>/templates/
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "myproject.wsgi.application"

# ── Database ───────────────────────────────────────────────────────────────────
# We only need sessions — SQLite is perfectly fine for that.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME"  : BASE_DIR / "db.sqlite3",
    }
}

# ── Sessions ───────────────────────────────────────────────────────────────────
# Store sessions in the database — simple and works out of the box.
SESSION_ENGINE = "django.contrib.sessions.backends.db"
SESSION_COOKIE_AGE = 3600 * 24  # 24 hours

# ── Static files ───────────────────────────────────────────────────────────────
STATIC_URL  = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# WhiteNoise: compresses and caches static files for efficiency
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# ── Internationalisation ───────────────────────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE     = "UTC"
USE_I18N      = True
USE_TZ        = True

# ── Logging ────────────────────────────────────────────────────────────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": os.environ.get("DJANGO_LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}

# ── Default primary key ────────────────────────────────────────────────────────
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── Security headers (important for Cloud Run) ────────────────────────────────
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT      = True
    SESSION_COOKIE_SECURE    = True
    CSRF_COOKIE_SECURE       = True