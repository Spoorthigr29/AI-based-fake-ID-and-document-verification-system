import os
import sys
from pathlib import Path
import django

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from django.test import Client
from django.contrib.auth.models import User

def run_auth_tests():
    client = Client()
    print("==================================================")
    print("VERIFYX AI - AUTHENTICATION SUITE TESTS")
    print("==================================================")

    # 1. Test GET /login/
    resp_get = client.get('/login/')
    print(f"1. GET /login/ status code: {resp_get.status_code}")
    assert resp_get.status_code == 200
    content = resp_get.content.decode('utf-8')
    assert "VerifyX AI" in content
    assert "Secure Document Verification System" in content
    assert "Email / Username" in content
    assert "Password" in content
    assert "Remember Me" in content
    assert "admin@verifyx.ai" in content
    assert "Admin@123" in content
    assert "Create Account" in content
    assert "Forgot Password" in content
    print("   [PASS] Sign In page renders all required titles, fields, links, and demo credentials.")

    # 2. Test Invalid Credentials
    resp_bad = client.post('/login/', {'username': 'admin@verifyx.ai', 'password': 'WrongPassword123'}, follow=True)
    assert resp_bad.status_code == 200
    assert "Invalid email or password." in resp_bad.content.decode('utf-8')
    print("   [PASS] Invalid credentials test returned: 'Invalid email or password.'")

    # 3. Test Valid Login with Email (admin@verifyx.ai / Admin@123)
    resp_login_email = client.post('/login/', {'username': 'admin@verifyx.ai', 'password': 'Admin@123', 'remember_me': 'true'}, follow=True)
    assert resp_login_email.status_code == 200
    # Check session
    assert '_auth_user_id' in client.session
    admin_user = User.objects.get(username='admin')
    assert int(client.session['_auth_user_id']) == admin_user.id
    print(f"   [PASS] Valid login with Email (admin@verifyx.ai): Authenticated user ID {client.session['_auth_user_id']}")

    # 4. Test Protected Route Access while logged in
    resp_analytics = client.get('/analytics/')
    print(f"4. Protected route /analytics/ status (logged in): {resp_analytics.status_code}")
    assert resp_analytics.status_code == 200
    print("   [PASS] Logged in user can access protected analytics and dashboard.")

    # 5. Test Logout
    resp_logout = client.get('/logout/', follow=True)
    print(f"5. GET /logout/ status code: {resp_logout.status_code}")
    assert resp_logout.status_code == 200
    assert '_auth_user_id' not in client.session
    print("   [PASS] Logout successful, session cleared and redirected to Sign In.")

    # 6. Test Protected Route Access when logged out
    resp_anon_analytics = client.get('/analytics/', follow=False)
    print(f"6. Protected route /analytics/ status (logged out): {resp_anon_analytics.status_code}")
    assert resp_anon_analytics.status_code == 302
    assert "/login/" in resp_anon_analytics.headers.get('Location', '')
    print("   [PASS] Unauthenticated access is properly blocked and redirected to /login/.")

    # 7. Test Login with Username (admin / Admin@123)
    resp_login_user = client.post('/login/', {'username': 'admin', 'password': 'Admin@123'}, follow=True)
    assert resp_login_user.status_code == 200
    assert '_auth_user_id' in client.session
    assert int(client.session['_auth_user_id']) == admin_user.id
    print("   [PASS] Valid login with Username (admin): Authenticated successfully.")

    print("\n==================================================")
    print("ALL AUTHENTICATION TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================")

if __name__ == '__main__':
    run_auth_tests()
