"""
VerifyX AI - Role-Based Access Control & Security Utilities
===========================================================
Defines fine-grained authorization helpers, client IP resolvers,
and document access control guards.
"""

import functools
from typing import List, Optional
from django.core.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from django.http import HttpResponseForbidden, JsonResponse
from rest_framework import permissions

from documents.models import Document
from audit.models import AuditLog


def get_client_ip(request) -> str:
    """Safely extract remote client IP address respecting reverse proxies."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
    return ip


def is_admin_user(user) -> bool:
    """Check if user has administrative privileges."""
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser or user.is_staff:
        return True
    profile = getattr(user, 'profile', None)
    return bool(profile and profile.is_admin())


def is_reviewer_user(user) -> bool:
    """Check if user is a reviewer or administrator."""
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser or user.is_staff:
        return True
    profile = getattr(user, 'profile', None)
    return bool(profile and profile.is_reviewer())


def can_access_document(user, document: Document, request=None) -> bool:
    """
    Evaluates whether the given user or request session has legitimate
    authorization to view or download document forensic artifacts.
    """
    # 1. System administrators and Reviewers have global case inspection access
    if is_admin_user(user) or is_reviewer_user(user):
        return True

    # 2. Authenticated user who uploaded the document
    if user and user.is_authenticated:
        if document.uploaded_by_id == user.id:
            return True
        return False

    # 3. Demo / Test records are accessible for presentation and testing
    if getattr(document, 'is_demo', False):
        return True

    # 4. Anonymous user in demo mode with session token matching verification ID or unassigned demo documents
    if request and hasattr(request, 'session'):
        allowed_ids = request.session.get('accessible_verifications', [])
        if document.verification_id in allowed_ids or str(document.id) in allowed_ids:
            return True
        if (not user or not user.is_authenticated) and document.uploaded_by is None:
            return True

    return False


def require_role(allowed_roles: List[str]):
    """
    View decorator enforcing institutional role authorization.
    """
    def decorator(view_func):
        @functools.wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                messages.warning(request, "Please authenticate to access restricted forensic consoles.")
                return redirect('accounts:login')

            user_role = 'USER'
            if is_admin_user(request.user):
                user_role = 'ADMIN'
            elif is_reviewer_user(request.user):
                user_role = 'REVIEWER'

            if user_role not in allowed_roles and not is_admin_user(request.user):
                AuditLog.log_event(
                    action=AuditLog.ACTION_UNAUTHORIZED_ATTEMPT,
                    user=request.user,
                    ip_address=get_client_ip(request),
                    metadata={"attempted_url": request.path, "required_roles": allowed_roles}
                )
                return HttpResponseForbidden("Access Denied: You do not possess the required security clearance.")

            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator


class IsDocumentAuthorized(permissions.BasePermission):
    """
    DRF Permission class ensuring only authorized users can view document endpoints.
    """
    def has_permission(self, request, view):
        return True

    def has_object_permission(self, request, view, obj):
        if isinstance(obj, Document):
            doc = obj
        elif hasattr(obj, 'document'):
            doc = obj.document
        else:
            return True

        has_access = can_access_document(request.user, doc, request)
        if not has_access:
            AuditLog.log_event(
                action=AuditLog.ACTION_UNAUTHORIZED_ATTEMPT,
                user=request.user,
                document=doc,
                ip_address=get_client_ip(request),
                metadata={"endpoint": request.path}
            )
        return has_access
