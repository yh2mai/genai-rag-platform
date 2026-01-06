import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  Card,
  CardContent,
  Stepper,
  Step,
  StepLabel,
  Divider,
} from '@mui/material';
import {
  CloudUpload,
  Description,
  CheckCircle,
  Error,
  Delete,
  Refresh,
  QuestionAnswer,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import { ProcessingResult, UploadProgress } from '../types';

const DocumentUpload: React.FC = () => {
  const navigate = useNavigate();
  const [uploads, setUploads] = useState<UploadProgress[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [hasCompletedUploads, setHasCompletedUploads] = useState(false);

  const steps = ['Upload Documents', 'Processing', 'Ask Questions'];
  const activeStep = hasCompletedUploads ? 2 : uploads.some(u => u.status === 'processing') ? 1 : 0;

  const validateFile = (file: File): string | null => {
    // Check file type
    if (file.type !== 'application/pdf') {
      return 'Only PDF files are supported';
    }
    
    // Check file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      return 'File size must be less than 50MB';
    }
    
    // Check filename
    if (file.name.length > 255) {
      return 'Filename is too long (max 255 characters)';
    }
    
    return null;
  };

  const uploadFile = async (file: File) => {
    // Add to uploads list
    const newUpload: UploadProgress = {
      filename: file.name,
      progress: 0,
      status: 'uploading',
    };
    
    setUploads(prev => [...prev, newUpload]);
    setError(null);
    setSuccess(null);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setUploads(prev => prev.map(upload => 
          upload.filename === file.name && upload.status === 'uploading'
            ? { ...upload, progress: Math.min(upload.progress + 10, 90) }
            : upload
        ));
      }, 200);

      const result: ProcessingResult = await apiService.uploadDocument(file);
      
      clearInterval(progressInterval);
      
      if (result.success) {
        setUploads(prev => prev.map(upload => 
          upload.filename === file.name
            ? { 
                ...upload, 
                progress: 100, 
                status: 'processing',
                message: 'Document uploaded successfully, processing...'
              }
            : upload
        ));

        // Simulate processing completion
        setTimeout(() => {
          setUploads(prev => prev.map(upload => 
            upload.filename === file.name && upload.status === 'processing'
              ? { 
                  ...upload, 
                  status: 'completed',
                  message: `Processed ${result.metadata?.page_count || 'unknown'} pages`
                }
              : upload
          ));
          setSuccess(`Successfully processed ${file.name}`);
          console.log("upload successfully")
          setHasCompletedUploads(true);
        }, 2000);
        
      } else {
        setUploads(prev => prev.map(upload => 
          upload.filename === file.name
            ? { 
                ...upload, 
                status: 'error',
                message: result.message || 'Upload failed'
              }
            : upload
        ));
        setError(`Failed to upload ${file.name}: ${result.message}`);
      }
    } catch (err: any) {
      setUploads(prev => prev.map(upload => 
        upload.filename === file.name
          ? { 
              ...upload, 
              status: 'error',
              message: err.message || 'Upload failed'
            }
          : upload
      ));
      setError(`Failed to upload ${file.name}: ${err.message || 'Unknown error'}`);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const validationError = validateFile(file);
      if (validationError) {
        setError(`${file.name}: ${validationError}`);
        continue;
      }
      
      await uploadFile(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const removeUpload = (filename: string) => {
    setUploads(prev => prev.filter(upload => upload.filename !== filename));
  };

  const retryUpload = async (filename: string) => {
    // Find the original file (this is a simplified retry - in real app you'd store the file)
    setError(`Retry functionality requires re-selecting the file: ${filename}`);
  };

  const getStatusColor = (status: UploadProgress['status']) => {
    switch (status) {
      case 'completed': return 'success';
      case 'error': return 'error';
      case 'processing': return 'info';
      case 'uploading': return 'primary';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: UploadProgress['status']) => {
    switch (status) {
      case 'completed': return <CheckCircle color="success" />;
      case 'error': return <Error color="error" />;
      case 'processing': return <Description color="info" />;
      case 'uploading': return <CloudUpload color="primary" />;
      default: return <Description />;
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Document Upload
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload PDF documents to add them to the knowledge base. 
        Supported formats: PDF (max 50MB per file)
      </Typography>

      {/* Workflow Stepper */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stepper activeStep={activeStep} alternativeLabel>
            {steps.map((label, index) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      {/* Upload Area */}
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          mb: 3,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
          cursor: 'pointer',
          textAlign: 'center',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: 'primary.main',
            backgroundColor: 'action.hover',
          },
        }}
      >
        <input {...getInputProps()} />
        <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive ? 'Drop files here' : 'Drag & drop PDF files here'}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          or click to select files
        </Typography>
        <Button variant="contained" component="span">
          Select Files
        </Button>
      </Paper>

      {/* Error Alert */}
      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          onClose={() => setError(null)}
        >
          {error}
        </Alert>
      )}

      {/* Success Alert with Next Steps */}
      {success && (
        <Alert 
          severity="success" 
          sx={{ mb: 2 }}
          onClose={() => setSuccess(null)}
          action={
            hasCompletedUploads && (
              <Button 
                color="inherit" 
                size="small"
                startIcon={<QuestionAnswer />}
                onClick={() => navigate('/query')}
                sx={{ ml: 2 }}
              >
                Ask Questions
              </Button>
            )
          }
        >
          {success}
          {hasCompletedUploads && (
            <Typography variant="body2" sx={{ mt: 1 }}>
              Your documents are ready! You can now ask questions about the uploaded content.
            </Typography>
          )}
        </Alert>
      )}

      {/* Upload Progress List */}
      {uploads.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Upload Progress
            </Typography>
            <List>
              {uploads.map((upload, index) => (
                <ListItem key={index} sx={{ px: 0 }}>
                  <ListItemIcon>
                    {getStatusIcon(upload.status)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body1">
                          {upload.filename}
                        </Typography>
                        <Chip 
                          label={upload.status} 
                          size="small" 
                          color={getStatusColor(upload.status)}
                        />
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        {upload.status === 'uploading' && (
                          <LinearProgress 
                            variant="determinate" 
                            value={upload.progress} 
                            sx={{ mb: 1 }}
                          />
                        )}
                        {upload.message && (
                          <Typography variant="body2" color="text.secondary">
                            {upload.message}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {upload.status === 'error' && (
                      <IconButton 
                        size="small" 
                        onClick={() => retryUpload(upload.filename)}
                        title="Retry upload"
                      >
                        <Refresh />
                      </IconButton>
                    )}
                    <IconButton 
                      size="small" 
                      onClick={() => removeUpload(upload.filename)}
                      title="Remove from list"
                    >
                      <Delete />
                    </IconButton>
                  </Box>
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      )}

      {/* Next Steps Card - shown when documents are completed */}
      {hasCompletedUploads && (
        <Card sx={{ mb: 3, backgroundColor: 'success.light', color: 'success.contrastText' }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <CheckCircle sx={{ mr: 2 }} />
              <Typography variant="h6">
                Documents Ready!
              </Typography>
            </Box>
            <Typography variant="body1" sx={{ mb: 3 }}>
              Your documents have been processed and are now available in the knowledge base. 
              You can start asking questions about the uploaded content.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                startIcon={<QuestionAnswer />}
                onClick={() => navigate('/query')}
                sx={{ 
                  backgroundColor: 'success.dark',
                  '&:hover': { backgroundColor: 'success.main' }
                }}
              >
                Start Asking Questions
              </Button>
              <Button
                variant="outlined"
                startIcon={<CloudUpload />}
                onClick={() => {
                  setUploads([]);
                  setHasCompletedUploads(false);
                  setSuccess(null);
                }}
                sx={{ 
                  borderColor: 'success.dark',
                  color: 'success.dark',
                  '&:hover': { 
                    borderColor: 'success.main',
                    backgroundColor: 'success.main',
                    color: 'white'
                  }
                }}
              >
                Upload More Documents
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Upload Guidelines */}
      <Paper sx={{ p: 3, backgroundColor: 'grey.50' }}>
        <Typography variant="h6" gutterBottom>
          Upload Guidelines
        </Typography>
        <List dense>
          <ListItem sx={{ px: 0 }}>
            <ListItemText primary="• Only PDF files are supported" />
          </ListItem>
          <ListItem sx={{ px: 0 }}>
            <ListItemText primary="• Maximum file size: 50MB" />
          </ListItem>
          <ListItem sx={{ px: 0 }}>
            <ListItemText primary="• Files will be processed automatically after upload" />
          </ListItem>
          <ListItem sx={{ px: 0 }}>
            <ListItemText primary="• Processing time depends on document size and complexity" />
          </ListItem>
          <ListItem sx={{ px: 0 }}>
            <ListItemText primary="• Uploaded documents will be available for querying once processed" />
          </ListItem>
        </List>
      </Paper>
    </Box>
  );
};

export default DocumentUpload;