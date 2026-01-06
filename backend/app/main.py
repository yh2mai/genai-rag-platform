"""
FastAPI main application entry point for Enterprise GenAI Platform
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.core.config import settings
from app.core.logging import setup_logging
from app.models.database import init_database

# Import API routers
from app.api.documents import router as documents_router
from app.api.queries import router as queries_router
from app.api.monitoring import router as monitoring_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize database
init_database()

# Create FastAPI application
app = FastAPI(
    title="Enterprise GenAI Platform API",
    description="Production-grade RAG system with agentic workflows",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(documents_router, prefix="/api/v1")
app.include_router(queries_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "Enterprise GenAI Platform API", "status": "running"}

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "enterprise-genai-platform"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )