"""
Database adapter to support both SQLite and MongoDB
"""
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Determine which database to use
USE_MONGODB = settings.DATABASE_TYPE.lower() == "mongodb"

if USE_MONGODB:
    from app.models.mongodb import (
        init_mongodb,
        MongoDocument,
        MongoChunk,
        MongoQueryLog,
        get_database as get_mongo_db
    )
    logger.info("Using MongoDB as database backend")
else:
    from app.models.database import (
        init_database,
        get_db as get_sqlite_db,
        Document as SQLDocument,
        DocumentChunk as SQLChunk,
        QueryLog as SQLQueryLog,
        SessionLocal
    )
    logger.info("Using SQLite as database backend")


def init_db():
    """Initialize database (works for both SQLite and MongoDB)"""
    if USE_MONGODB:
        init_mongodb()
    else:
        init_database()


def get_db() -> Generator:
    """Get database session/connection (works for both SQLite and MongoDB)"""
    if USE_MONGODB:
        # For MongoDB, we don't need a session, but we yield the database for consistency
        db = get_mongo_db()
        try:
            yield db
        finally:
            pass  # MongoDB doesn't need explicit session closing
    else:
        # For SQLite, use the existing session management
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()


class DocumentAdapter:
    """Unified document operations for both databases"""
    
    @staticmethod
    def create(document_data: Dict[str, Any]) -> str:
        """Create a new document"""
        if USE_MONGODB:
            return MongoDocument.create(document_data)
        else:
            db = SessionLocal()
            try:
                doc = SQLDocument(**document_data)
                db.add(doc)
                db.commit()
                db.refresh(doc)
                return doc.document_id
            finally:
                db.close()
    
    @staticmethod
    def get_by_id(document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if USE_MONGODB:
            doc = MongoDocument.get_by_id(document_id)
            if doc:
                doc.pop('_id', None)  # Remove MongoDB internal ID
            return doc
        else:
            db = SessionLocal()
            try:
                doc = db.query(SQLDocument).filter(SQLDocument.document_id == document_id).first()
                if doc:
                    return {
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "file_path": doc.file_path,
                        "file_size": doc.file_size,
                        "page_count": doc.page_count,
                        "creation_date": doc.creation_date,
                        "processing_date": doc.processing_date,
                        "checksum": doc.checksum,
                        "document_metadata": doc.document_metadata,
                        "processing_status": doc.processing_status,
                        "created_at": doc.created_at,
                        "updated_at": doc.updated_at
                    }
                return None
            finally:
                db.close()
    
    @staticmethod
    def get_by_checksum(checksum: str) -> Optional[Dict[str, Any]]:
        """Get document by checksum"""
        if USE_MONGODB:
            doc = MongoDocument.get_by_checksum(checksum)
            if doc:
                doc.pop('_id', None)
            return doc
        else:
            db = SessionLocal()
            try:
                doc = db.query(SQLDocument).filter(SQLDocument.checksum == checksum).first()
                if doc:
                    return {"document_id": doc.document_id, "filename": doc.filename}
                return None
            finally:
                db.close()
    
    @staticmethod
    def get_all(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all documents with pagination"""
        if USE_MONGODB:
            docs = MongoDocument.get_all(skip, limit)
            for doc in docs:
                doc.pop('_id', None)
            return docs
        else:
            db = SessionLocal()
            try:
                docs = db.query(SQLDocument).order_by(SQLDocument.created_at.desc()).offset(skip).limit(limit).all()
                return [
                    {
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "file_size": doc.file_size,
                        "page_count": doc.page_count,
                        "processing_status": doc.processing_status,
                        "created_at": doc.created_at,
                        "checksum": doc.checksum
                    }
                    for doc in docs
                ]
            finally:
                db.close()
    
    @staticmethod
    def count() -> int:
        """Count total documents"""
        if USE_MONGODB:
            return MongoDocument.count()
        else:
            db = SessionLocal()
            try:
                return db.query(SQLDocument).count()
            finally:
                db.close()
    
    @staticmethod
    def update(document_id: str, update_data: Dict[str, Any]) -> bool:
        """Update document"""
        if USE_MONGODB:
            return MongoDocument.update(document_id, update_data)
        else:
            db = SessionLocal()
            try:
                doc = db.query(SQLDocument).filter(SQLDocument.document_id == document_id).first()
                if doc:
                    for key, value in update_data.items():
                        setattr(doc, key, value)
                    doc.updated_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
            finally:
                db.close()
    
    @staticmethod
    def delete(document_id: str) -> bool:
        """Delete document"""
        if USE_MONGODB:
            return MongoDocument.delete(document_id)
        else:
            db = SessionLocal()
            try:
                doc = db.query(SQLDocument).filter(SQLDocument.document_id == document_id).first()
                if doc:
                    db.delete(doc)
                    db.commit()
                    return True
                return False
            finally:
                db.close()


class ChunkAdapter:
    """Unified chunk operations for both databases"""
    
    @staticmethod
    def create_many(chunks_data: List[Dict[str, Any]]) -> List[str]:
        """Create multiple chunks"""
        if USE_MONGODB:
            return MongoChunk.create_many(chunks_data)
        else:
            db = SessionLocal()
            try:
                chunk_ids = []
                for chunk_data in chunks_data:
                    chunk = SQLChunk(**chunk_data)
                    db.add(chunk)
                    chunk_ids.append(chunk.chunk_id)
                db.commit()
                return chunk_ids
            finally:
                db.close()
    
    @staticmethod
    def get_by_document(document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        if USE_MONGODB:
            chunks = MongoChunk.get_by_document(document_id)
            for chunk in chunks:
                chunk.pop('_id', None)
            return chunks
        else:
            db = SessionLocal()
            try:
                chunks = db.query(SQLChunk).filter(SQLChunk.document_id == document_id).order_by(SQLChunk.chunk_index).all()
                return [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "char_count": chunk.char_count,
                        "embedding_id": chunk.embedding_id
                    }
                    for chunk in chunks
                ]
            finally:
                db.close()
    
    @staticmethod
    def count() -> int:
        """Count total chunks"""
        if USE_MONGODB:
            return MongoChunk.count()
        else:
            db = SessionLocal()
            try:
                return db.query(SQLChunk).count()
            finally:
                db.close()


class QueryLogAdapter:
    """Unified query log operations for both databases"""
    
    @staticmethod
    def create(query_data: Dict[str, Any]) -> str:
        """Create a new query log"""
        if USE_MONGODB:
            return MongoQueryLog.create(query_data)
        else:
            db = SessionLocal()
            try:
                query_log = SQLQueryLog(**query_data)
                db.add(query_log)
                db.commit()
                db.refresh(query_log)
                return query_log.query_id
            finally:
                db.close()
    
    @staticmethod
    def get_by_id(query_id: str) -> Optional[Dict[str, Any]]:
        """Get query log by ID"""
        if USE_MONGODB:
            log = MongoQueryLog.get_by_id(query_id)
            if log:
                log.pop('_id', None)
            return log
        else:
            db = SessionLocal()
            try:
                log = db.query(SQLQueryLog).filter(SQLQueryLog.query_id == query_id).first()
                if log:
                    return {
                        "query_id": log.query_id,
                        "user_query": log.user_query,
                        "response": log.response,
                        "citations": log.citations,
                        "processing_time": log.processing_time,
                        "token_usage": log.token_usage,
                        "evaluation_metrics": log.evaluation_metrics,
                        "timestamp": log.timestamp
                    }
                return None
            finally:
                db.close()
    
    @staticmethod
    def get_all(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all query logs with pagination"""
        if USE_MONGODB:
            logs = MongoQueryLog.get_all(skip, limit)
            for log in logs:
                log.pop('_id', None)
            return logs
        else:
            db = SessionLocal()
            try:
                logs = db.query(SQLQueryLog).order_by(SQLQueryLog.timestamp.desc()).offset(skip).limit(limit).all()
                return [
                    {
                        "query_id": log.query_id,
                        "user_query": log.user_query,
                        "response": log.response,
                        "citations": log.citations,
                        "processing_time": log.processing_time,
                        "token_usage": log.token_usage,
                        "evaluation_metrics": log.evaluation_metrics,
                        "timestamp": log.timestamp
                    }
                    for log in logs
                ]
            finally:
                db.close()
    
    @staticmethod
    def count() -> int:
        """Count total query logs"""
        if USE_MONGODB:
            return MongoQueryLog.count()
        else:
            db = SessionLocal()
            try:
                return db.query(SQLQueryLog).count()
            finally:
                db.close()
