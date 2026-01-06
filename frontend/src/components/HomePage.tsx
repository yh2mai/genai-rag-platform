import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Paper,
} from '@mui/material';
import {
  CloudUpload,
  Search,
  Dashboard,
  Description,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: 'Document Upload',
      description: 'Upload and process PDF documents for knowledge extraction',
      icon: <CloudUpload sx={{ fontSize: 40 }} />,
      path: '/upload',
      color: '#1976d2',
    },
    {
      title: 'Query Interface',
      description: 'Ask questions and get AI-powered answers with citations',
      icon: <Search sx={{ fontSize: 40 }} />,
      path: '/query',
      color: '#388e3c',
    },
    {
      title: 'Admin Dashboard',
      description: 'Monitor system performance and manage configurations',
      icon: <Dashboard sx={{ fontSize: 40 }} />,
      path: '/admin',
      color: '#f57c00',
    },
  ];

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Paper
        elevation={0}
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          p: 6,
          mb: 4,
          borderRadius: 2,
        }}
      >
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Welcome to Enterprise GenAI Platform
        </Typography>
        <Typography variant="h6" align="center" sx={{ opacity: 0.9 }}>
          Intelligent document processing with RAG and agentic workflows
        </Typography>
      </Paper>

      <Grid container spacing={4}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 4,
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', p: 3 }}>
                <Box
                  sx={{
                    color: feature.color,
                    mb: 2,
                  }}
                >
                  {feature.icon}
                </Box>
                <Typography variant="h5" component="h2" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  {feature.description}
                </Typography>
                <Button
                  variant="contained"
                  onClick={() => navigate(feature.path)}
                  sx={{
                    backgroundColor: feature.color,
                    '&:hover': {
                      backgroundColor: feature.color,
                      opacity: 0.8,
                    },
                  }}
                >
                  Get Started
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 6 }}>
        <Paper sx={{ p: 4 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8}>
              <Typography variant="h5" gutterBottom>
                System Status
              </Typography>
              <Typography variant="body1" color="text.secondary">
                The Enterprise GenAI Platform is ready for document processing and intelligent querying.
                All core services are operational and ready to handle your knowledge management needs.
              </Typography>
            </Grid>
            <Grid item xs={12} md={4} sx={{ textAlign: 'center' }}>
              <Description sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" color="primary">
                Ready to Process
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Box>
  );
};

export default HomePage;