"""
VerifyX AI - Clear Demo History Command
=======================================
Safely removes ONLY records created with is_demo=True.
Never deletes real user screening records or model files.
"""

from django.core.management.base import BaseCommand
from documents.models import Document


class Command(BaseCommand):
    help = "Safely remove only DEMO/TEST screening records (is_demo=True), preserving all real verification records"

    def handle(self, *args, **options):
        demo_docs = Document.objects.filter(is_demo=True)
        count = demo_docs.count()

        if count == 0:
            self.stdout.write(self.style.WARNING("No DEMO screening records found to delete."))
        else:
            # Delete files associated with demo docs safely
            for doc in demo_docs:
                try:
                    if doc.original_file:
                        doc.original_file.delete(save=False)
                    if doc.selfie_file:
                        doc.selfie_file.delete(save=False)
                except Exception:
                    pass

            deleted_count, _ = demo_docs.delete()
            self.stdout.write(self.style.SUCCESS(
                f"[OK] Successfully cleared {count} DEMO screening records."
            ))

        remaining_real = Document.objects.filter(is_demo=False).count()
        self.stdout.write(self.style.SUCCESS(
            f"Preserved {remaining_real} real screening records in the database."
        ))
