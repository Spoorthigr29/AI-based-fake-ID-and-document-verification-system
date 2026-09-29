from django.contrib import admin
from .models import AuditLog

@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'action', 'user', 'document')
    list_filter = ('action', 'timestamp')
    search_fields = ('user__username', 'document__verification_id', 'action')
    readonly_fields = ('user', 'action', 'document', 'timestamp', 'metadata')
