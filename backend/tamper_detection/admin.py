from django.contrib import admin
from .models import TamperAnalysis

@admin.register(TamperAnalysis)
class TamperAnalysisAdmin(admin.ModelAdmin):
    list_display = ('document', 'risk_level', 'tampering_probability', 'created_at')
    list_filter = ('risk_level', 'created_at')
    search_fields = ('document__verification_id',)
