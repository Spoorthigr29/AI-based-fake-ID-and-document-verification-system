from django.shortcuts import render
from .models import AuditLog

def audit_trail_view(request):
    """View compliance audit trail and activity log."""
    logs = AuditLog.objects.select_related('user', 'document').order_by('-timestamp')[:100]
    return render(request, 'audit/trail.html', {'logs': logs})
