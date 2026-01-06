import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Send,
  ExpandMore,
  ContentCopy,
  ThumbUp,
  ThumbDown,
  History,
  Clear,
  CloudUpload,
  Warning,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import { QueryRequest, QueryResponse, Citation } from '../types';

interface QueryHistoryItem {
  id: string;
  query: string;
  response: QueryResponse;
  timestamp: Date;
}

const QueryInterface: React.FC = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [queryHistory, setQueryHistory] = useState<QueryHistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [documentsAvailable, setDocumentsAvailable] = useState<boolean | null>(null);
  
  const queryInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Focus on query input when component mounts
    if (queryInputRef.current) {
      queryInputRef.current.focus();
    }
    
    // Check if documents are available
    checkDocumentsAvailability();
  }, []);

  const checkDocumentsAvailability = async () => {
    try {
      const documents = await apiService.getDocuments();
      setDocumentsAvailable(documents.length > 0);
    } catch (error) {
      console.warn('Could not check document availability:', error);
      setDocumentsAvailable(null); // Unknown state
    }
  };

  const handleSubmitQuery = async (queryText?: string) => {
    const currentQuery = queryText || query;
    if (!currentQuery.trim()) return;

    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const queryRequest: QueryRequest = {
        query: currentQuery,
        max_results: 10,
        include_citations: true,
      };

      const result = await apiService.submitQuery(queryRequest);
      setResponse(result);
      
      // Add to history
      const historyItem: QueryHistoryItem = {
        id: `${Date.now()}-${Math.random()}`,
        query: currentQuery,
        response: result,
        timestamp: new Date(),
      };
      
      setQueryHistory(prev => [historyItem, ...prev.slice(0, 9)]); // Keep last 10 queries
      
      if (!queryText) {
        setQuery(''); // Clear input only if not from history
      }
    } catch (err: any) {
      setError(err.message || 'Failed to process query');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmitQuery();
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatConfidenceScore = (score: number): string => {
    return `${Math.round(score * 100)}%`;
  };

  const getConfidenceColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const renderCitation = (citation: Citation, index: number) => (
    <Card key={index} variant="outlined" sx={{ mb: 1 }}>
      <CardContent sx={{ py: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
          <Typography variant="subtitle2" color="primary">
            {citation.filename} (Page {citation.page_number})
          </Typography>
          <Chip 
            label={`${Math.round(citation.relevance_score * 100)}%`}
            size="small"
            color={citation.relevance_score > 0.7 ? 'success' : 'default'}
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          "{citation.chunk_content.substring(0, 200)}..."
        </Typography>
        <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
          <Tooltip title="Copy citation">
            <IconButton 
              size="small" 
              onClick={() => copyToClipboard(citation.chunk_content)}
            >
              <ContentCopy fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </CardContent>
    </Card>
  );

  const renderQueryHistory = () => (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Query History
          </Typography>
          <Button
            startIcon={<Clear />}
            onClick={() => setQueryHistory([])}
            size="small"
          >
            Clear History
          </Button>
        </Box>
        <List>
          {queryHistory.map((item, index) => (
            <React.Fragment key={item.id}>
              <ListItem 
                sx={{ 
                  cursor: 'pointer',
                  '&:hover': { backgroundColor: 'action.hover' }
                }}
                onClick={() => handleSubmitQuery(item.query)}
              >
                <ListItemIcon>
                  <History />
                </ListItemIcon>
                <ListItemText
                  primary={item.query}
                  secondary={`${item.timestamp.toLocaleString()} • Confidence: ${formatConfidenceScore(item.response.confidence_score)}`}
                />
              </ListItem>
              {index < queryHistory.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
        {queryHistory.length === 0 && (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
            No queries yet. Start by asking a question above.
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  return (
    <Box sx={{ maxWidth: 1000, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Query Interface
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Ask questions about your uploaded documents and get AI-powered answers with citations.
      </Typography>

      {/* No Documents Warning */}
      {documentsAvailable === false && (
        <Alert 
          severity="warning" 
          sx={{ mb: 3 }}
          icon={<Warning />}
          action={
            <Button 
              color="inherit" 
              size="small"
              startIcon={<CloudUpload />}
              onClick={() => navigate('/upload')}
            >
              Upload Documents
            </Button>
          }
        >
          <Typography variant="subtitle2" gutterBottom>
            No documents available for querying
          </Typography>
          <Typography variant="body2">
            You need to upload and process documents before you can ask questions. 
            Upload PDF documents to build your knowledge base.
          </Typography>
        </Alert>
      )}

      {/* Query Input */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
          <TextField
            ref={queryInputRef}
            fullWidth
            multiline
            maxRows={4}
            label="Ask a question..."
            placeholder={documentsAvailable === false 
              ? "Upload documents first to start asking questions" 
              : "e.g., What are the key findings in the research papers?"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading || documentsAvailable === false}
            variant="outlined"
          />
          <Button
            variant="contained"
            onClick={() => handleSubmitQuery()}
            disabled={isLoading || !query.trim() || documentsAvailable === false}
            startIcon={isLoading ? <CircularProgress size={20} /> : <Send />}
            sx={{ minWidth: 120, height: 56 }}
          >
            {isLoading ? 'Processing...' : 'Ask'}
          </Button>
        </Box>
        
        <Box sx={{ mt: 2, display: 'flex', gap: 1, alignItems: 'center' }}>
          <Button
            size="small"
            startIcon={<History />}
            onClick={() => setShowHistory(!showHistory)}
            disabled={documentsAvailable === false}
          >
            {showHistory ? 'Hide' : 'Show'} History ({queryHistory.length})
          </Button>
          {documentsAvailable === false && (
            <Typography variant="caption" color="text.secondary">
              Upload documents to enable querying
            </Typography>
          )}
        </Box>
      </Paper>

      {/* Error Alert */}
      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 3 }}
          onClose={() => setError(null)}
        >
          {error}
        </Alert>
      )}

      {/* Query History */}
      {showHistory && renderQueryHistory()}

      {/* Response */}
      {response && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Answer
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <Chip 
                  label={`Confidence: ${formatConfidenceScore(response.confidence_score)}`}
                  color={getConfidenceColor(response.confidence_score)}
                  size="small"
                />
                <Chip 
                  label={`${response.processing_time.toFixed(2)}s`}
                  variant="outlined"
                  size="small"
                />
              </Box>
            </Box>
            
            <Typography variant="body1" sx={{ mb: 2, lineHeight: 1.6 }}>
              {response.answer}
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
              <Tooltip title="Copy answer">
                <IconButton 
                  size="small" 
                  onClick={() => copyToClipboard(response.answer)}
                >
                  <ContentCopy />
                </IconButton>
              </Tooltip>
              <Tooltip title="Helpful">
                <IconButton size="small">
                  <ThumbUp />
                </IconButton>
              </Tooltip>
              <Tooltip title="Not helpful">
                <IconButton size="small">
                  <ThumbDown />
                </IconButton>
              </Tooltip>
            </Box>

            {/* Citations */}
            {response.citations && response.citations.length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="subtitle1">
                    Sources & Citations ({response.citations.length})
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ mt: 1 }}>
                    {response.citations.map((citation, index) => 
                      renderCitation(citation, index)
                    )}
                  </Box>
                </AccordionDetails>
              </Accordion>
            )}
          </CardContent>
        </Card>
      )}

      {/* Sample Queries or Upload Prompt */}
      {!response && !isLoading && (
        <Paper sx={{ p: 3, backgroundColor: 'grey.50' }}>
          {documentsAvailable === false ? (
            <>
              <Typography variant="h6" gutterBottom>
                Get Started
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                To start asking questions, you need to upload documents first. Here's how:
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h6" color="primary">1.</Typography>
                  <Typography variant="body1">
                    Upload PDF documents using the Document Upload page
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h6" color="primary">2.</Typography>
                  <Typography variant="body1">
                    Wait for documents to be processed (usually takes a few seconds)
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="h6" color="primary">3.</Typography>
                  <Typography variant="body1">
                    Return here to ask questions about your documents
                  </Typography>
                </Box>
                <Box sx={{ mt: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={<CloudUpload />}
                    onClick={() => navigate('/upload')}
                    size="large"
                  >
                    Upload Your First Document
                  </Button>
                </Box>
              </Box>
            </>
          ) : (
            <>
              <Typography variant="h6" gutterBottom>
                Sample Queries
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Try these example questions to get started:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {[
                  "What are the main topics covered in the documents?",
                  "Summarize the key findings from the research",
                  "What recommendations are mentioned?",
                  "Are there any specific dates or deadlines mentioned?",
                  "What are the main conclusions?",
                ].map((sampleQuery, index) => (
                  <Chip
                    key={index}
                    label={sampleQuery}
                    onClick={() => setQuery(sampleQuery)}
                    clickable
                    variant="outlined"
                    sx={{ mb: 1 }}
                  />
                ))}
              </Box>
            </>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default QueryInterface;