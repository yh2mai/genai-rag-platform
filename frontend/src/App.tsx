import React from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import { Container, AppBar, Toolbar, Typography, Box } from '@mui/material';
import HomePage from './components/HomePage';
import DocumentUpload from './components/DocumentUpload';
import QueryInterface from './components/QueryInterface';
import AdminDashboard from './components/AdminDashboard';
import ErrorBoundary from './components/ErrorBoundary';
import { NotificationProvider } from './components/NotificationSystem';
import './App.css';

function App() {
  const navigate = useNavigate();

  const handleTitleClick = () => {
    navigate('/');
  };

  return (
    <NotificationProvider>
      <ErrorBoundary>
        <Box sx={{ flexGrow: 1, height: '100vh', display: 'flex', flexDirection: 'column' }}>
          <AppBar position="static">
            <Toolbar>
              <Typography 
                variant="h6" 
                component="div" 
                sx={{ 
                  flexGrow: 1,
                  cursor: 'pointer',
                  '&:hover': {
                    opacity: 0.8,
                  },
                }}
                onClick={handleTitleClick}
              >
                Enterprise GenAI Platform
              </Typography>
            </Toolbar>
          </AppBar>
          
          <Container 
            maxWidth="lg" 
            sx={{ 
              flexGrow: 1, 
              display: 'flex', 
              flexDirection: 'column',
              py: 3 
            }}
          >
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/upload" element={<DocumentUpload />} />
              <Route path="/query" element={<QueryInterface />} />
              <Route path="/admin" element={<AdminDashboard />} />
            </Routes>
          </Container>
        </Box>
      </ErrorBoundary>
    </NotificationProvider>
  );
}

export default App;