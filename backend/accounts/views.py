from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
from django.utils.http import url_has_allowed_host_and_scheme
from django.conf import settings
from django.views.decorators.csrf import csrf_protect

@csrf_protect
def login_view(request):
    """
    Professional VerifyX AI Sign In View.
    Supports email or username authentication, remember-me sessions,
    and demo instant sign-in with admin@verifyx.ai / Admin@123.
    """
    next_url = request.GET.get('next') or request.POST.get('next') or 'dashboard:home'
    
    # If user is already logged in, redirect them directly
    if request.user.is_authenticated:
        if url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
            return redirect(next_url)
        return redirect('dashboard:home')

    # Ensure default demonstration admin user exists
    _ensure_demo_user()

    if request.method == 'POST':
        # 1. Quick Demo Sign-In Action
        if 'demo_login' in request.POST:
            admin_user = User.objects.filter(email__iexact='admin@verifyx.ai').first() or User.objects.filter(username='admin').first()
            if not admin_user:
                admin_user = _ensure_demo_user()
            login(request, admin_user)
            messages.success(request, "Authenticated successfully with demonstration credentials.")
            if url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
                return redirect(next_url)
            return redirect('dashboard:home')

        # 2. Standard Email / Username Authentication
        identifier = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        remember_me = request.POST.get('remember_me')

        if not identifier or not password:
            messages.error(request, "Invalid email or password.")
            return render(request, 'accounts/login.html', {
                'username': identifier,
                'next': next_url,
            })

        # Resolve username if email address was provided
        auth_username = identifier
        if '@' in identifier:
            matched_user = User.objects.filter(email__iexact=identifier).first()
            if matched_user:
                auth_username = matched_user.username

        # Authenticate user with hashed password check
        user = authenticate(request, username=auth_username, password=password)

        # Fallback: if username lookup failed, check if identifier was username directly
        if user is None and '@' not in identifier:
            matched_by_email = User.objects.filter(email__iexact=identifier).first()
            if matched_by_email:
                user = authenticate(request, username=matched_by_email.username, password=password)

        if user is not None:
            if not user.is_active:
                messages.error(request, "This account is inactive. Please contact your system administrator.")
                return render(request, 'accounts/login.html', {'username': identifier, 'next': next_url})

            login(request, user)

            # Handle Remember Me session expiration
            if remember_me:
                request.session.set_expiry(1209600)  # 2 weeks
            else:
                request.session.set_expiry(0)  # Expires when browser closes

            messages.success(request, f"Welcome back, {user.get_full_name() or user.username}!")
            
            # Safe redirect validation
            if next_url and url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
                return redirect(next_url)
            return redirect('dashboard:home')
        else:
            messages.error(request, "Invalid email or password.")
            return render(request, 'accounts/login.html', {
                'username': identifier,
                'next': next_url,
            })

    return render(request, 'accounts/login.html', {'next': next_url})


def logout_view(request):
    """
    Secure Sign Out View.
    Terminates the user session and redirects to the sign-in portal.
    """
    logout(request)
    messages.info(request, "You have been securely signed out of VerifyX AI.")
    return redirect('accounts:login')


@csrf_protect
def register_view(request):
    """
    Create Account Portal for new VerifyX users.
    """
    if request.user.is_authenticated:
        return redirect('dashboard:home')

    if request.method == 'POST':
        full_name = request.POST.get('full_name', '').strip()
        email = request.POST.get('email', '').strip().lower()
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        confirm_password = request.POST.get('confirm_password', '').strip()

        if not username:
            username = email.split('@')[0] if email else ''

        if not email or not password or not username:
            messages.error(request, "All fields are required to create an account.")
            return render(request, 'accounts/register.html', {
                'full_name': full_name, 'email': email, 'username': username
            })

        if password != confirm_password:
            messages.error(request, "Passwords do not match. Please re-enter.")
            return render(request, 'accounts/register.html', {
                'full_name': full_name, 'email': email, 'username': username
            })

        if len(password) < 6:
            messages.error(request, "Password must be at least 6 characters long.")
            return render(request, 'accounts/register.html', {
                'full_name': full_name, 'email': email, 'username': username
            })

        if User.objects.filter(username__iexact=username).exists():
            messages.error(request, f"Username '{username}' is already taken. Please select another.")
            return render(request, 'accounts/register.html', {
                'full_name': full_name, 'email': email, 'username': username
            })

        if User.objects.filter(email__iexact=email).exists():
            messages.error(request, "An account with this email address already exists. Please sign in.")
            return redirect('accounts:login')

        # Create user with properly hashed password
        first_name = full_name.split()[0] if full_name else ''
        last_name = ' '.join(full_name.split()[1:]) if len(full_name.split()) > 1 else ''
        
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )
        login(request, user)
        messages.success(request, f"Account created successfully! Welcome to VerifyX AI, {user.first_name or user.username}.")
        return redirect('dashboard:home')

    return render(request, 'accounts/register.html')


def forgot_password_view(request):
    """
    Forgot Password / Account Recovery view.
    """
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        messages.info(request, f"If an account exists for {email}, recovery instructions have been dispatched. For immediate institutional access, please contact admin@verifyx.ai.")
        return redirect('accounts:login')
    return render(request, 'accounts/forgot_password.html')


def profile_view(request):
    """
    User Profile & Security Settings View.
    """
    if not request.user.is_authenticated:
        return redirect('accounts:login')
    return render(request, 'accounts/profile.html')


def _ensure_demo_user():
    """
    Ensures the demo administrator account exists with the exact requested credentials:
    Email: admin@verifyx.ai
    Password: Admin@123
    """
    admin_user = User.objects.filter(username='admin').first()
    if not admin_user:
        admin_user = User.objects.create_superuser('admin', 'admin@verifyx.ai', 'Admin@123')
    else:
        admin_user.email = 'admin@verifyx.ai'
        admin_user.set_password('Admin@123')
        admin_user.save()
    return admin_user
