from django.apps import AppConfig

class RagConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag'
    verbose_name = 'VerifyX RAG Knowledge & Explanation Engine'

    def ready(self):
        """Pre-initialize or load the vector index when Django starts."""
        try:
            from .services.rag_service import RAGService
            # Lazy warm-up without blocking server startup
            rag = RAGService.get_instance()
            rag.initialize()
        except Exception:
            pass
