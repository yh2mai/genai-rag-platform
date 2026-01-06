// API service for Enterprise GenAI Platform
import axios from 'axios';
import { QueryRequest, QueryResponse, SystemHealth, ProcessingResult } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    
    // Enhance error with user-friendly messages
    if (error.code === 'ECONNABORTED') {
      error.message = 'Request timeout. The server is taking too long to respond.';
    } else if (error.code === 'ERR_NETWORK') {
      error.message = 'Network error. Please check your internet connection.';
    } else if (!error.response) {
      error.message = 'Unable to connect to the server. Please try again later.';
    }
    
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async getHealth(): Promise<SystemHealth> {
    try {
      const response = await apiClient.get('/api/v1/monitoring/health');
      const healthData = response.data;
      
      // Transform backend health response to frontend format
      return {
        status: healthData.status,
        services: {
          api: healthData.components?.database?.status === 'healthy',
          vector_db: healthData.components?.vector_store?.status === 'healthy',
          document_store: healthData.components?.storage?.status === 'healthy',
          embedding_service: healthData.components?.vector_store?.status === 'healthy',
        },
        uptime: healthData.uptime_seconds || 0,
      };
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Unable to connect to backend server');
    }
  },

  // Document upload
  async uploadDocument(file: File): Promise<ProcessingResult> {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await apiClient.post('/api/v1/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // Longer timeout for file uploads
      });
      
      const uploadData = response.data;

      return {
        success: uploadData.success,
        document_id: uploadData.document_id,
        message: uploadData.message,
        metadata: {
          document_id: uploadData.document_id,
          filename: file.name,
          file_size: file.size,
          page_count: 0, // Will be updated after processing
          creation_date: new Date().toISOString(),
          processing_date: new Date().toISOString(),
          checksum: 'processing',
        },
      };
    } catch (error: any) {
      console.error('Document upload failed:', error);
      throw new Error(error.response?.data?.detail || 'Failed to upload document');
    }
  },

  // Query processing
  async submitQuery(queryRequest: QueryRequest): Promise<QueryResponse> {
    try {
      // Submit query for async processing
      const submitResponse = await apiClient.post('/api/v1/queries/', {
        query: queryRequest.query,
        max_chunks: queryRequest.max_results || 5,
        use_agents: false, // Use RAG pipeline for reliable processing
        include_citations: queryRequest.include_citations !== false,
      });
      
      const queryId = submitResponse.data.query_id;
      console.log(`Query submitted with ID: ${queryId}`);
      
      // Poll for results
      let attempts = 0;
      const maxAttempts = 60; // 60 seconds timeout for better reliability
      
      while (attempts < maxAttempts) {
        try {
          const resultResponse = await apiClient.get(`/api/v1/queries/${queryId}`);
          const result = resultResponse.data;
          
          if (result.status === 'completed') {
            console.log(`Query ${queryId} completed successfully`);
            return {
              answer: result.response,
              citations: result.citations || [],
              processing_time: result.processing_time,
              confidence_score: result.evaluation_score || 0.8,
            };
          } else if (result.status === 'failed') {
            console.error(`Query ${queryId} failed`);
            throw new Error('Query processing failed');
          }
          
          // Log progress if available
          if (attempts % 5 === 0) {
            console.log(`Query ${queryId} still processing... (attempt ${attempts}/${maxAttempts})`);
          }
          
          // Wait 1 second before next attempt
          await new Promise(resolve => setTimeout(resolve, 1000));
          attempts++;
        } catch (pollError) {
          console.warn(`Error polling query result (attempt ${attempts}):`, pollError);
          attempts++;
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      
      throw new Error('Query processing timeout - please try again');
      
    } catch (error: any) {
      console.error('Query processing failed:', error);
      
      // Return a helpful error response instead of throwing
      return {
        answer: `I apologize, but I'm unable to process your query at the moment due to a backend service issue. The query "${queryRequest.query}" could not be processed. Please try again later or contact support if the issue persists.`,
        citations: [],
        processing_time: 0,
        confidence_score: 0,
      };
    }
  },

  // Get document list
  async getDocuments(): Promise<any[]> {
    try {
      const response = await apiClient.get('/api/v1/documents/');
      return response.data.documents || [];
    } catch (error) {
      console.error('Failed to fetch documents:', error);
      throw new Error('Unable to fetch documents from server');
    }
  },

  // Get system metrics
  async getMetrics(): Promise<any> {
    try {
      const response = await apiClient.get('/api/v1/monitoring/dashboard');
      const dashboardData = response.data;
      
      return {
        total_documents: dashboardData.database_metrics.total_documents,
        total_queries: dashboardData.database_metrics.total_queries,
        avg_response_time: dashboardData.performance_metrics.average_query_time,
        total_tokens_used: dashboardData.performance_metrics.total_token_usage.total || 0,
        estimated_cost: dashboardData.performance_metrics.total_cost_usd,
        cache_hit_rate: dashboardData.performance_metrics.cache_hit_rate,
        system_uptime: dashboardData.system_metrics.cpu_usage_percent,
        active_users: dashboardData.performance_metrics.queries_per_hour,
      };
    } catch (error) {
      console.warn('Dashboard API not fully available, fetching individual metrics');
      
      try {
        // Try to get basic database metrics
        const dbResponse = await apiClient.get('/api/v1/monitoring/metrics/database');
        const dbMetrics = dbResponse.data;
        
        return {
          total_documents: dbMetrics.total_documents || 0,
          total_queries: dbMetrics.total_queries || 0,
          avg_response_time: dbMetrics.average_processing_time || 0,
          total_tokens_used: 0,
          estimated_cost: 0,
          cache_hit_rate: 0,
          system_uptime: 0,
          active_users: 0,
        };
      } catch (fallbackError) {
        console.error('Failed to fetch any metrics:', fallbackError);
        throw new Error('Unable to fetch system metrics from server');
      }
    }
  },
};

export default apiService;