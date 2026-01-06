// Mock API service for testing
import { QueryRequest, QueryResponse, SystemHealth, ProcessingResult } from '../../types';

export const apiService = {
  async getHealth(): Promise<SystemHealth> {
    return {
      status: 'healthy',
      services: {
        api: true,
        vector_db: true,
        document_store: true,
        embedding_service: true,
      },
      uptime: 3600,
    };
  },

  async uploadDocument(file: File): Promise<ProcessingResult> {
    return {
      success: true,
      document_id: 'test_doc_123',
      message: 'Document uploaded successfully',
      metadata: {
        document_id: 'test_doc_123',
        filename: file.name,
        file_size: file.size,
        page_count: 5,
        creation_date: new Date().toISOString(),
        processing_date: new Date().toISOString(),
        checksum: 'test_checksum',
      },
    };
  },

  async submitQuery(queryRequest: QueryRequest): Promise<QueryResponse> {
    return {
      answer: `Mock answer for: ${queryRequest.query}`,
      citations: [
        {
          document_id: 'test_doc_123',
          filename: 'test_document.pdf',
          page_number: 1,
          chunk_content: 'This is a test citation.',
          relevance_score: 0.85,
        },
      ],
      processing_time: 1.2,
      confidence_score: 0.9,
    };
  },

  async getDocuments(): Promise<any[]> {
    return [
      {
        document_id: 'test_doc_123',
        filename: 'test_document.pdf',
        upload_date: new Date().toISOString(),
      },
    ];
  },

  async getMetrics(): Promise<any> {
    return {
      total_documents: 5,
      total_queries: 25,
      avg_response_time: 1.2,
      total_tokens_used: 10000,
      estimated_cost: 5.50,
      cache_hit_rate: 0.75,
      system_uptime: 3600,
      active_users: 3,
    };
  },
};

export default apiService;