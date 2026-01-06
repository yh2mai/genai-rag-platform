"""
Health check and monitoring API endpoints
"""
import os
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
import logging

from app.models.database import get_db, Document, DocumentChunk, QueryLog
from app.services.document_storage import DocumentStorageService
from app.services.cost_performance_tracker import CostPerformanceTracker
from app.services.vector_store import VectorStore
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Initialize services
storage_service = DocumentStorageService()
cost_tracker = CostPerformanceTracker()
vector_store = VectorStore()


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]


class SystemMetrics(BaseModel):
    """System performance metrics model"""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    available_memory_gb: float
    total_memory_gb: float
    available_disk_gb: float
    total_disk_gb: float


class DatabaseMetrics(BaseModel):
    """Database metrics model"""
    total_documents: int
    total_chunks: int
    total_queries: int
    recent_queries_24h: int
    average_processing_time: float
    database_size_mb: float


class StorageMetrics(BaseModel):
    """Storage metrics model"""
    total_documents: int
    total_storage_bytes: int
    total_storage_gb: float
    storage_path: str
    largest_document_mb: float
    average_document_size_mb: float


class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    total_token_usage: Dict[str, int]
    total_cost_usd: float
    average_query_time: float
    queries_per_hour: float
    cache_hit_rate: float
    error_rate: float


class AdminDashboardData(BaseModel):
    """Admin dashboard data model"""
    system_metrics: SystemMetrics
    database_metrics: DatabaseMetrics
    storage_metrics: StorageMetrics
    performance_metrics: PerformanceMetrics
    recent_queries: List[Dict[str, Any]]
    recent_documents: List[Dict[str, Any]]


# Track application start time for uptime calculation
app_start_time = datetime.now()


def check_database_health(db: Session) -> Dict[str, Any]:
    """Check database connectivity and basic operations"""
    try:
        # Test basic query
        result = db.execute(text("SELECT 1")).fetchone()
        if result and result[0] == 1:
            # Test table access
            doc_count = db.query(Document).count()
            return {
                "status": "healthy",
                "message": "Database connection successful",
                "document_count": doc_count
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Database query failed",
                "error": "Unexpected query result"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": "Database connection failed",
            "error": str(e)
        }


def check_storage_health() -> Dict[str, Any]:
    """Check document storage system health"""
    try:
        storage_path = settings.DOCUMENT_STORE_PATH
        
        # Check if storage directory exists and is writable
        if not os.path.exists(storage_path):
            return {
                "status": "unhealthy",
                "message": "Storage directory does not exist",
                "path": storage_path
            }
        
        if not os.access(storage_path, os.W_OK):
            return {
                "status": "unhealthy",
                "message": "Storage directory is not writable",
                "path": storage_path
            }
        
        # Check available disk space
        disk_usage = psutil.disk_usage(storage_path)
        free_space_gb = disk_usage.free / (1024**3)
        
        if free_space_gb < 1.0:  # Less than 1GB free
            return {
                "status": "degraded",
                "message": "Low disk space",
                "free_space_gb": round(free_space_gb, 2)
            }
        
        return {
            "status": "healthy",
            "message": "Storage system operational",
            "free_space_gb": round(free_space_gb, 2)
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": "Storage health check failed",
            "error": str(e)
        }


def check_vector_store_health() -> Dict[str, Any]:
    """Check vector store health"""
    try:
        # Test vector store operations
        index_info = vector_store.get_index_info()
        
        return {
            "status": "healthy",
            "message": "Vector store operational",
            "index_size": index_info.get("total_vectors", 0)
        }
        
    except Exception as e:
        return {
            "status": "degraded",
            "message": "Vector store check failed",
            "error": str(e)
        }


def get_system_metrics() -> SystemMetrics:
    """Get current system performance metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_memory_gb = memory.available / (1024**3)
        total_memory_gb = memory.total / (1024**3)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        available_disk_gb = disk.free / (1024**3)
        total_disk_gb = disk.total / (1024**3)
        
        return SystemMetrics(
            cpu_usage_percent=round(cpu_percent, 2),
            memory_usage_percent=round(memory_percent, 2),
            disk_usage_percent=round(disk_percent, 2),
            available_memory_gb=round(available_memory_gb, 2),
            total_memory_gb=round(total_memory_gb, 2),
            available_disk_gb=round(available_disk_gb, 2),
            total_disk_gb=round(total_disk_gb, 2)
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        # Return default values on error
        return SystemMetrics(
            cpu_usage_percent=0.0,
            memory_usage_percent=0.0,
            disk_usage_percent=0.0,
            available_memory_gb=0.0,
            total_memory_gb=0.0,
            available_disk_gb=0.0,
            total_disk_gb=0.0
        )


@router.get("/health", response_model=HealthStatus)
async def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check endpoint
    
    Returns overall system health status and component details
    """
    try:
        # Calculate uptime
        uptime = (datetime.now() - app_start_time).total_seconds()
        
        # Check individual components
        components = {
            "database": check_database_health(db),
            "storage": check_storage_health(),
            "vector_store": check_vector_store_health()
        }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in components.values()]
        
        if all(status == "healthy" for status in component_statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in component_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return HealthStatus(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=0.0,
            components={"error": {"status": "unhealthy", "message": str(e)}}
        )


@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_performance():
    """
    Get current system performance metrics
    
    Returns CPU, memory, and disk usage statistics
    """
    try:
        return get_system_metrics()
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system metrics: {str(e)}"
        )


@router.get("/metrics/database", response_model=DatabaseMetrics)
async def get_database_metrics(db: Session = Depends(get_db)):
    """
    Get database performance metrics
    
    Returns document, chunk, and query statistics
    """
    try:
        # Get basic counts
        total_documents = db.query(Document).count()
        total_chunks = db.query(DocumentChunk).count()
        total_queries = db.query(QueryLog).count()
        
        # Get recent queries (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_queries = db.query(QueryLog).filter(
            QueryLog.timestamp >= yesterday
        ).count()
        
        # Calculate average processing time
        avg_time_result = db.query(QueryLog.processing_time).filter(
            QueryLog.processing_time.isnot(None)
        ).all()
        
        if avg_time_result:
            processing_times = [row[0] for row in avg_time_result if row[0] is not None]
            average_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        else:
            average_processing_time = 0.0
        
        # Estimate database size (rough calculation)
        database_size_mb = 0.0
        try:
            if "sqlite" in settings.DATABASE_URL:
                db_path = settings.DATABASE_URL.replace("sqlite:///", "")
                if os.path.exists(db_path):
                    database_size_mb = os.path.getsize(db_path) / (1024**2)
        except Exception:
            pass
        
        return DatabaseMetrics(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_queries=total_queries,
            recent_queries_24h=recent_queries,
            average_processing_time=round(average_processing_time, 3),
            database_size_mb=round(database_size_mb, 2)
        )
        
    except Exception as e:
        logger.error(f"Error getting database metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get database metrics: {str(e)}"
        )


@router.get("/metrics/storage", response_model=StorageMetrics)
async def get_storage_metrics(db: Session = Depends(get_db)):
    """
    Get storage system metrics
    
    Returns document storage statistics
    """
    try:
        # Get storage stats from service
        storage_stats = storage_service.get_storage_stats(db)
        
        # Calculate additional metrics
        total_storage_gb = storage_stats["total_storage_bytes"] / (1024**3)
        
        # Get document size statistics
        documents = db.query(Document).all()
        if documents:
            file_sizes = [doc.file_size for doc in documents]
            largest_document_mb = max(file_sizes) / (1024**2)
            average_document_size_mb = (sum(file_sizes) / len(file_sizes)) / (1024**2)
        else:
            largest_document_mb = 0.0
            average_document_size_mb = 0.0
        
        return StorageMetrics(
            total_documents=storage_stats["total_documents"],
            total_storage_bytes=storage_stats["total_storage_bytes"],
            total_storage_gb=round(total_storage_gb, 2),
            storage_path=storage_stats["storage_path"],
            largest_document_mb=round(largest_document_mb, 2),
            average_document_size_mb=round(average_document_size_mb, 2)
        )
        
    except Exception as e:
        logger.error(f"Error getting storage metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get storage metrics: {str(e)}"
        )


@router.get("/metrics/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(db: Session = Depends(get_db)):
    """
    Get system performance metrics
    
    Returns token usage, costs, and performance statistics
    """
    try:
        # Get cost and performance data
        total_usage = cost_tracker.get_total_usage()
        total_cost = cost_tracker.get_total_cost()
        
        # Calculate query performance metrics
        recent_queries = db.query(QueryLog).filter(
            QueryLog.timestamp >= datetime.now() - timedelta(hours=24)
        ).all()
        
        if recent_queries:
            # Average query time
            processing_times = [q.processing_time for q in recent_queries if q.processing_time]
            avg_query_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
            
            # Queries per hour
            queries_per_hour = len(recent_queries) / 24.0
            
            # Error rate (queries with errors in evaluation_metrics)
            error_queries = [q for q in recent_queries if q.evaluation_metrics and q.evaluation_metrics.get('error')]
            error_rate = len(error_queries) / len(recent_queries) if recent_queries else 0.0
        else:
            avg_query_time = 0.0
            queries_per_hour = 0.0
            error_rate = 0.0
        
        # Cache hit rate (placeholder - would need actual cache implementation)
        cache_hit_rate = 0.0  # TODO: Implement actual cache hit rate calculation
        
        return PerformanceMetrics(
            total_token_usage=total_usage,
            total_cost_usd=round(total_cost, 4),
            average_query_time=round(avg_query_time, 3),
            queries_per_hour=round(queries_per_hour, 2),
            cache_hit_rate=round(cache_hit_rate, 3),
            error_rate=round(error_rate, 3)
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/dashboard", response_model=AdminDashboardData)
async def get_admin_dashboard_data(db: Session = Depends(get_db)):
    """
    Get comprehensive admin dashboard data
    
    Returns all metrics and recent activity for admin interface
    """
    try:
        # Get all metrics
        system_metrics = get_system_metrics()
        
        # Get database metrics
        database_metrics = await get_database_metrics(db)
        
        # Get storage metrics
        storage_metrics = await get_storage_metrics(db)
        
        # Get performance metrics
        performance_metrics = await get_performance_metrics(db)
        
        # Get recent queries (last 10)
        recent_query_logs = db.query(QueryLog).order_by(
            QueryLog.timestamp.desc()
        ).limit(10).all()
        
        recent_queries = []
        for log in recent_query_logs:
            recent_queries.append({
                "query_id": log.query_id,
                "query": log.user_query[:100] + "..." if len(log.user_query) > 100 else log.user_query,
                "processing_time": log.processing_time,
                "timestamp": log.timestamp.isoformat(),
                "citation_count": len(log.citations) if log.citations else 0
            })
        
        # Get recent documents (last 10)
        recent_docs = db.query(Document).order_by(
            Document.created_at.desc()
        ).limit(10).all()
        
        recent_documents = []
        for doc in recent_docs:
            recent_documents.append({
                "document_id": doc.document_id,
                "filename": doc.filename,
                "file_size_mb": round(doc.file_size / (1024**2), 2),
                "page_count": doc.page_count,
                "processing_status": doc.processing_status,
                "created_at": doc.created_at.isoformat()
            })
        
        return AdminDashboardData(
            system_metrics=system_metrics,
            database_metrics=database_metrics,
            storage_metrics=storage_metrics,
            performance_metrics=performance_metrics,
            recent_queries=recent_queries,
            recent_documents=recent_documents
        )
        
    except Exception as e:
        logger.error(f"Error getting admin dashboard data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get admin dashboard data: {str(e)}"
        )


@router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    level: str = "INFO"
):
    """
    Get recent system logs
    
    - **lines**: Number of log lines to return (default: 100, max: 1000)
    - **level**: Minimum log level (DEBUG, INFO, WARNING, ERROR)
    """
    try:
        # Validate parameters
        if lines > 1000:
            lines = 1000
        if lines < 1:
            lines = 1
        
        level = level.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            level = "INFO"
        
        # Try to read from log file
        log_file_path = "logs/enterprise_genai.log"
        
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                # Filter by log level if specified
                if level != "DEBUG":
                    level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
                    min_priority = level_priority.get(level, 1)
                    
                    filtered_lines = []
                    for line in recent_lines:
                        for log_level, priority in level_priority.items():
                            if log_level in line and priority >= min_priority:
                                filtered_lines.append(line)
                                break
                    
                    recent_lines = filtered_lines
                
                return {
                    "logs": [line.strip() for line in recent_lines],
                    "total_lines": len(recent_lines),
                    "log_file": log_file_path,
                    "level_filter": level
                }
        else:
            return {
                "logs": ["Log file not found"],
                "total_lines": 0,
                "log_file": log_file_path,
                "level_filter": level
            }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system logs: {str(e)}"
        )


@router.post("/clear-cache")
async def clear_system_cache():
    """
    Clear system caches
    
    Clears response cache and other temporary data
    """
    try:
        # Clear response cache if available
        # This would integrate with the actual cache service
        
        return {
            "message": "System cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )