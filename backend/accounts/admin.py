from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'department', 'created_at', 'updated_at')
    list_filter = ('role', 'department', 'created_at')
    search_fields = ('user__username', 'user__email', 'department')
