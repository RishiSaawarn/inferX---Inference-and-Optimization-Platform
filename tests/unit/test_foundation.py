from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "env": settings.app_env}
    
def test_config_loaded():
    # Verify that the config has some basic values loaded
    assert settings.app_env is not None
    assert settings.log_level is not None
