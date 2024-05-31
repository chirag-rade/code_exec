from django.test import TestCase, Client as TestClient
from django.urls import reverse

class TenantAccessTestCase(TestCase):
    def setUp(self):
        self.client = TestClient()

    def test_non_existent_tenant_access(self):
        response = self.client.get('/', HTTP_HOST='nonexistent.example.com')
        print(f"Response status code: {response.status_code}")
        self.assertEqual(response.status_code, 404)

# Running the test
if __name__ == "__main__":
    import django
    from django.conf import settings
    from django.core.management import execute_from_command_line

    settings.configure(
        DATABASES={
            'default': {
                'ENGINE': 'django_tenant_schemas.postgresql_backend',
                'NAME': 'your_db_name',
                'USER': 'your_db_user',
                'PASSWORD': 'your_db_password',
                'HOST': 'localhost',
                'PORT': '',
            }
        },
        INSTALLED_APPS=[
            'django_tenant_schemas',
            'myapp',
            'django.contrib.contenttypes',
            'django.contrib.auth',
            'django.contrib.sessions',
            'django.contrib.messages',
            'django.contrib.staticfiles',
        ],
        MIDDLEWARE=[
            'django_tenant_schemas.middleware.TenantMiddleware',
            'django.middleware.common.CommonMiddleware',
            'django.contrib.sessions.middleware.SessionMiddleware',
            'django.middleware.csrf.CsrfViewMiddleware',
            'django.contrib.auth.middleware.AuthenticationMiddleware',
            'django.contrib.messages.middleware.MessageMiddleware',
            'django.middleware.clickjacking.XFrameOptionsMiddleware',
        ],
        TENANT_MODEL="myapp.Client",
        TENANT_DOMAIN_MODEL="myapp.Domain",
        ROOT_URLCONF=__name__,
    )

    try:
        django.setup()
        execute_from_command_line(['manage.py', 'test', 'myapp.tests.TenantAccessTestCase'])
    except Exception as e:
        print(f"Exception occurred: {e}")
