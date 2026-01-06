import React from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Refresh,
  Home,
  ContactSupport,
  CheckCircle,
} from '@mui/icons-material';

interface ErrorFallbackProps {
  error?: Error;
  resetError?: () => void;
  title?: string;
  message?: string;
  showTechnicalDetails?: boolean;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetError,
  title = "Something went wrong",
  message = "We encountered an unexpected error. Please try the suggestions below.",
  showTechnicalDetails = false,
}) => {
  const handleGoHome = () => {
    window.location.href = '/';
  };

  const handleReload = () => {
    window.location.reload();
  };

  const handleContactSupport = () => {
    // In a real app, this would open a support ticket or email
    window.open('mailto:support@example.com?subject=Error Report', '_blank');
  };

  const troubleshootingSteps = [
    {
      icon: <Refresh />,
      primary: "Refresh the page",
      secondary: "Sometimes a simple refresh can resolve temporary issues",
      action: handleReload,
    },
    {
      icon: <Home />,
      primary: "Go to homepage",
      secondary: "Return to the main page and try navigating again",
      action: handleGoHome,
    },
    {
      icon: <CheckCircle />,
      primary: "Check your internet connection",
      secondary: "Ensure you have a stable internet connection",
    },
    {
      icon: <ContactSupport />,
      primary: "Contact support",
      secondary: "If the problem persists, please contact our support team",
      action: handleContactSupport,
    },
  ];

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '50vh',
        p: 3,
      }}
    >
      <Paper
        sx={{
          p: 4,
          maxWidth: 600,
          width: '100%',
        }}
      >
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <ErrorIcon
            sx={{
              fontSize: 64,
              color: 'error.main',
              mb: 2,
            }}
          />
          
          <Typography variant="h5" gutterBottom>
            {title}
          </Typography>
          
          <Typography variant="body1" color="text.secondary">
            {message}
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            <Typography variant="body2">
              {error.message || 'An unknown error occurred'}
            </Typography>
          </Alert>
        )}

        <Typography variant="h6" gutterBottom>
          What you can try:
        </Typography>

        <List>
          {troubleshootingSteps.map((step, index) => (
            <ListItem
              key={index}
              sx={{
                cursor: step.action ? 'pointer' : 'default',
                '&:hover': step.action ? { backgroundColor: 'action.hover' } : {},
                borderRadius: 1,
              }}
              onClick={step.action}
            >
              <ListItemIcon sx={{ color: 'primary.main' }}>
                {step.icon}
              </ListItemIcon>
              <ListItemText
                primary={step.primary}
                secondary={step.secondary}
              />
            </ListItem>
          ))}
        </List>

        {resetError && (
          <Box sx={{ textAlign: 'center', mt: 3 }}>
            <Button
              variant="contained"
              onClick={resetError}
              startIcon={<Refresh />}
            >
              Try Again
            </Button>
          </Box>
        )}

        {showTechnicalDetails && error && process.env.NODE_ENV === 'development' && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Technical Details (Development Mode):
            </Typography>
            <Typography
              variant="body2"
              component="pre"
              sx={{
                backgroundColor: 'grey.100',
                p: 2,
                borderRadius: 1,
                overflow: 'auto',
                fontSize: '0.75rem',
                maxHeight: 200,
              }}
            >
              {error.stack}
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default ErrorFallback;