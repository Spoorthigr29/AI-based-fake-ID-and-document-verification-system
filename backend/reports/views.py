from django.shortcuts import render
from django.http import JsonResponse
from .models import VerificationReport

def reports_list_view(request):
    """View list of generated verification reports."""
    reports = VerificationReport.objects.select_related('document', 'generated_by').order_by('-created_at')
    return render(request, 'reports/list.html', {'reports': reports})
