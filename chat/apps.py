from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"

    def ready(self):
        import sys
        if "runserver" in sys.argv or "gunicorn" in sys.argv[0]:
            from gpt.loader import initialise_model
            try:
                initialise_model()
            except Exception as e:
                print(f"[ChatConfig] ⚠️  Could not load model at startup: {e}")
                print("[ChatConfig]    Model will be loaded on first request.")