from django.contrib import admin
from .models import RiskFactor

@admin.register(RiskFactor)
class RiskFactorAdmin(admin.ModelAdmin):
    list_display = ('factor_name', 'factor_value', 'contribution', 'verification_result')
    list_filter = ('factor_name',)
    search_fields = ('factor_name', 'description', 'verification_result__document__verification_id')
