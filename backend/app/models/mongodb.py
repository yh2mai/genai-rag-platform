"""
MongoDB models and database connection for document storage
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global MongoDB client
_mongo_client: Optional[MongoClient] = None
_database = None


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client"""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    return _mongo_client


def get_database():
    """Get MongoDB database instance"""
    global _database
    if _database is None:
        client = get_mongo_client()
        _database = client[settings.MONGODB_DATABASE]
    return _database


def init_mongodb():
    """Initialize MongoDB collections and indexes"""
    try:
        db = get_database()
        
        # Create collections if they don't exist
        collections = db.list_collection_names()
        
        if "documents" not in collections:
            db.create_collection("documents")
        
        if "chunks" not in collections:
            db.create_collection("chunks")
        
        if "query_logs" not in collections:
            db.create_collection("query_logs")
        
        # Create indexes for documents collection
        db.documents.create_index([("document_id", ASCENDING)], unique=True)
        db.documents.create_index([("checksum", ASCENDING)], unique=True)
        db.documents.create_index([("created_at", DESCENDING)])
        db.documents.create_index([("processing_status", ASCENDING)])
        
        # Create indexes for chunks collection
        db.chunks.create_index([("chunk_id", ASCENDING)], unique=True)
        db.chunks.create_index([("document_id", ASCENDING)])
        db.chunks.create_index([("document_id", ASCENDING), ("chunk_index", ASCENDING)])
        
        # Create indexes for query_logs collection
        db.query_logs.create_index([("query_id", ASCENDING)], unique=True)
        db.query_logs.create_index([("timestamp", DESCENDING)])
        
        logger.info("MongoDB collections and indexes initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {e}")
        raise


class MongoDocument:
    """MongoDB Document model"""
    
    @staticmethod
    def create(document_data: Dict[str, Any]) -> str:
        """Create a new document"""
        db = get_database()
        
        # Add timestamps
        document_data["created_at"] = datetime.utcnow()
        document_data["updated_at"] = datetime.utcnow()
        
        # Set default values
        if "processing_status" not in document_data:
            document_data["processing_status"] = "completed"
        
        try:
            result = db.documents.insert_one(document_data)
            return str(result.inserted_id)
        except DuplicateKeyError:
            raise ValueError("Document with this ID or checksum already exists")
    
    @staticmethod
    def get_by_id(document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        db = get_database()
        return db.documents.find_one({"document_id": document_id})
    
    @staticmethod
    def get_by_checksum(checksum: str) -> Optional[Dict[str, Any]]:
        """Get document by checksum"""
        db = get_database()
        return db.documents.find_one({"checksum": checksum})
    
    @staticmethod
    def get_all(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all documents with pagination"""
        db = get_database()
        return list(db.documents.find().sort("created_at", DESCENDING).skip(skip).limit(limit))
    
    @staticmethod
    def count() -> int:
        """Count total documents"""
        db = get_database()
        return db.documents.count_documents({})
    
    @staticmethod
    def update(document_id: str, update_data: Dict[str, Any]) -> bool:
        """Update document"""
        db = get_database()
        update_data["updated_at"] = datetime.utcnow()
        result = db.documents.update_one(
            {"document_id": document_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    @staticmethod
    def delete(document_id: str) -> bool:
        """Delete document and its chunks"""
        db = get_database()
        
        # Delete chunks first
        db.chunks.delete_many({"document_id": document_id})
        
        # Delete document
        result = db.documents.delete_one({"document_id": document_id})
        return result.deleted_count > 0


class MongoChunk:
    """MongoDB Chunk model"""
    
    @staticmethod
    def create(chunk_data: Dict[str, Any]) -> str:
        """Create a new chunk"""
        db = get_database()
        chunk_data["created_at"] = datetime.utcnow()
        
        try:
            result = db.chunks.insert_one(chunk_data)
            return str(result.inserted_id)
        except DuplicateKeyError:
            raise ValueError("Chunk with this ID already exists")
    
    @staticmethod
    def create_many(chunks_data: List[Dict[str, Any]]) -> List[str]:
        """Create multiple chunks"""
        db = get_database()
        
        for chunk in chunks_data:
            chunk["created_at"] = datetime.utcnow()
        
        result = db.chunks.insert_many(chunks_data)
        return [str(id) for id in result.inserted_ids]
    
    @staticmethod
    def get_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk by ID"""
        db = get_database()
        return db.chunks.find_one({"chunk_id": chunk_id})
    
    @staticmethod
    def get_by_document(document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        db = get_database()
        return list(db.chunks.find({"document_id": document_id}).sort("chunk_index", ASCENDING))
    
    @staticmethod
    def count() -> int:
        """Count total chunks"""
        db = get_database()
        return db.chunks.count_documents({})
    
    @staticmethod
    def delete_by_document(document_id: str) -> int:
        """Delete all chunks for a document"""
        db = get_database()
        result = db.chunks.delete_many({"document_id": document_id})
        return result.deleted_count


class MongoQueryLog:
    """MongoDB QueryLog model"""
    
    @staticmethod
    def create(query_data: Dict[str, Any]) -> str:
        """Create a new query log"""
        db = get_database()
        query_data["timestamp"] = datetime.utcnow()
        
        try:
            result = db.query_logs.insert_one(query_data)
            return str(result.inserted_id)
        except DuplicateKeyError:
            raise ValueError("Query log with this ID already exists")
    
    @staticmethod
    def get_by_id(query_id: str) -> Optional[Dict[str, Any]]:
        """Get query log by ID"""
        db = get_database()
        return db.query_logs.find_one({"query_id": query_id})
    
    @staticmethod
    def get_recent(limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent query logs"""
        db = get_database()
        return list(db.query_logs.find().sort("timestamp", DESCENDING).limit(limit))
    
    @staticmethod
    def get_all(skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all query logs with pagination"""
        db = get_database()
        return list(db.query_logs.find().sort("timestamp", DESCENDING).skip(skip).limit(limit))
    
    @staticmethod
    def count() -> int:
        """Count total query logs"""
        db = get_database()
        return db.query_logs.count_documents({})
    
    @staticmethod
    def delete(query_id: str) -> bool:
        """Delete query log"""
        db = get_database()
        result = db.query_logs.delete_one({"query_id": query_id})
        return result.deleted_count > 0
    
    @staticmethod
    def get_statistics() -> Dict[str, Any]:
        """Get query statistics"""
        db = get_database()
        
        pipeline = [
            {
                "$match": {"processing_time": {"$ne": None}}
            },
            {
                "$group": {
                    "_id": None,
                    "total_queries": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$processing_time"},
                    "min_processing_time": {"$min": "$processing_time"},
                    "max_processing_time": {"$max": "$processing_time"}
                }
            }
        ]
        
        result = list(db.query_logs.aggregate(pipeline))
        
        if result:
            return result[0]
        else:
            return {
                "total_queries": 0,
                "avg_processing_time": 0,
                "min_processing_time": 0,
                "max_processing_time": 0
            }


def close_mongodb():
    """Close MongoDB connection"""
    global _mongo_client, _database
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        _database = None
        logger.info("MongoDB connection closed")
