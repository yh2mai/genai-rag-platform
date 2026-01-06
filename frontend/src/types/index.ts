// Type definitions for Enterprise GenAI Platform

export interface DocumentMetadata {
  document_id: string;
  filename: string;
  file_size: number;
  page_count: number;
  creation_date: string;
  processing_date: string;
  checksum: string;
}

export interface ProcessingResult {
  success: boolean;
  document_id?: string;
  message: string;
  metadata?: DocumentMetadata;
}

export interface QueryRequest {
  query: string;
  max_results?: number;
  include_citations?: boolean;
}

export interface Citation {
  document_id: string;
  filename: string;
  page_number: number;
  chunk_content: string;
  relevance_score: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  processing_time: number;
  confidence_score: number;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    api: boolean;
    vector_db: boolean;
    document_store: boolean;
    embedding_service: boolean;
  };
  uptime: number;
}

export interface UploadProgress {
  filename: string;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  message?: string;
}