"""
Database models and schema for document storage
"""
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

from app.core.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    document_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)
    creation_date = Column(DateTime, nullable=True)
    processing_date = Column(DateTime, default=datetime.utcnow)
    checksum = Column(String, nullable=False, unique=True)
    document_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata'
    processing_status = Column(String, default="completed")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Document chunks table"""
    __tablename__ = "chunks"
    
    chunk_id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.document_id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=False)
    embedding_id = Column(String, nullable=True)  # Reference to vector database
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to document
    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    """Query logs table for monitoring and evaluation"""
    __tablename__ = "query_logs"
    
    query_id = Column(String, primary_key=True, index=True)
    user_query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    citations = Column(JSON, nullable=True)
    processing_time = Column(Float, nullable=True)
    token_usage = Column(JSON, nullable=True)
    evaluation_metrics = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize database and create tables"""
    # Ensure database directory exists
    if "sqlite" in settings.DATABASE_URL:
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    # Create tables
    create_tables()