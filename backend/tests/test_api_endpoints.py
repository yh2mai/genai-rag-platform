"""
Tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

from app.main import app
from app.models.database import Base, get_db


# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module")
def setup_database():
    """Set up test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Clean up test database file
    if os.path.exists("test.db"):
        os.unlink("test.db")


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Enterprise GenAI Platform API"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test basic health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "enterprise-genai-platform"


def test_comprehensive_health_endpoint(setup_database):
    """Test comprehensive health check endpoint"""
    response = client.get("/api/v1/monitoring/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data
    assert "uptime_seconds" in data
    assert "components" in data


def test_system_metrics_endpoint(setup_database):
    """Test system metrics endpoint"""
    response = client.get("/api/v1/monitoring/metrics/system")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_usage_percent" in data
    assert "memory_usage_percent" in data
    assert "disk_usage_percent" in data


def test_database_metrics_endpoint(setup_database):
    """Test database metrics endpoint"""
    response = client.get("/api/v1/monitoring/metrics/database")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "total_queries" in data


def test_document_list_endpoint(setup_database):
    """Test document listing endpoint"""
    response = client.get("/api/v1/documents/")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert "skip" in data
    assert "limit" in data
    assert isinstance(data["documents"], list)


def test_query_history_endpoint(setup_database):
    """Test query history endpoint"""
    response = client.get("/api/v1/queries/")
    assert response.status_code == 200
    data = response.json()
    assert "queries" in data
    assert "total" in data
    assert "skip" in data
    assert "limit" in data
    assert isinstance(data["queries"], list)


def test_document_upload_validation():
    """Test document upload validation"""
    # Test with non-PDF file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"This is a test file")
        temp_file.flush()
        
        with open(temp_file.name, "rb") as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        os.unlink(temp_file.name)
    
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_query_submission_validation():
    """Test query submission validation"""
    # Test with empty query
    response = client.post(
        "/api/v1/queries/",
        json={"query": ""}
    )
    assert response.status_code == 422  # Validation error
    
    # Test with valid query
    response = client.post(
        "/api/v1/queries/",
        json={"query": "What is machine learning?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert "status" in data


def test_direct_query_processing(setup_database):
    """Test direct query processing endpoint"""
    response = client.post(
        "/api/v1/queries/direct",
        json={
            "query": "What is artificial intelligence?",
            "max_chunks": 3,
            "use_agents": False,
            "include_citations": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert "query" in data
    assert "response" in data
    assert "citations" in data
    assert "processing_time" in data
    assert "status" in data


def test_nonexistent_document():
    """Test getting nonexistent document"""
    response = client.get("/api/v1/documents/nonexistent-id")
    assert response.status_code == 404


def test_nonexistent_query():
    """Test getting nonexistent query"""
    response = client.get("/api/v1/queries/nonexistent-id")
    assert response.status_code == 404


def test_admin_dashboard_endpoint(setup_database):
    """Test admin dashboard data endpoint"""
    response = client.get("/api/v1/monitoring/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "system_metrics" in data
    assert "database_metrics" in data
    assert "storage_metrics" in data
    assert "performance_metrics" in data
    assert "recent_queries" in data
    assert "recent_documents" in data


if __name__ == "__main__":
    pytest.main([__file__])