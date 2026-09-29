from django.contrib import admin
from .models import VerificationReport

@admin.register(VerificationReport)
class VerificationReportAdmin(admin.ModelAdmin):
    list_display = ('document', 'report_format', 'generated_by', 'created_at')
    list_filter = ('report_format', 'created_at')
    search_fields = ('document__verification_id', 'summary')
