import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Box,
  Chip,
} from '@mui/material';
import {
  ExpandMore,
  Wifi,
  Refresh,
  Storage,
  CloudOff,
  BugReport,
  ContactSupport,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
} from '@mui/icons-material';

interface ErrorRecoveryGuideProps {
  open: boolean;
  onClose: () => void;
  errorType?: 'network' | 'server' | 'validation' | 'upload' | 'query' | 'general';
  errorMessage?: string;
}

const ErrorRecoveryGuide: React.FC<ErrorRecoveryGuideProps> = ({
  open,
  onClose,
  errorType = 'general',
  errorMessage,
}) => {
  const getErrorTypeInfo = (type: string) => {
    switch (type) {
      case 'network':
        return {
          title: 'Network Connection Issues',
          icon: <Wifi color="error" />,
          color: 'error' as const,
          description: 'Problems connecting to the server',
        };
      case 'server':
        return {
          title: 'Server Error',
          icon: <CloudOff color="error" />,
          color: 'error' as const,
          description: 'The server encountered an error',
        };
      case 'validation':
        return {
          title: 'Input Validation Error',
          icon: <Warning color="warning" />,
          color: 'warning' as const,
          description: 'There was an issue with the provided input',
        };
      case 'upload':
        return {
          title: 'File Upload Error',
          icon: <Storage color="error" />,
          color: 'error' as const,
          description: 'Problem uploading your file',
        };
      case 'query':
        return {
          title: 'Query Processing Error',
          icon: <BugReport color="error" />,
          color: 'error' as const,
          description: 'Error processing your query',
        };
      default:
        return {
          title: 'General Error',
          icon: <ErrorIcon color="error" />,
          color: 'error' as const,
          description: 'An unexpected error occurred',
        };
    }
  };

  const getRecoverySteps = (type: string): Array<{
    icon: React.ReactElement;
    title: string;
    description: string;
    action?: () => void;
  }> => {
    const commonSteps = [
      {
        icon: <Refresh />,
        title: 'Refresh the page',
        description: 'Try reloading the page to reset the application state',
        action: () => window.location.reload(),
      },
      {
        icon: <ContactSupport />,
        title: 'Contact support',
        description: 'If the problem persists, reach out to our support team',
        action: () => window.open('mailto:support@example.com', '_blank'),
      },
    ];

    switch (type) {
      case 'network':
        return [
          {
            icon: <Wifi />,
            title: 'Check your internet connection',
            description: 'Ensure you have a stable internet connection',
          },
          {
            icon: <Refresh />,
            title: 'Try again in a few moments',
            description: 'Network issues are often temporary',
          },
          ...commonSteps,
        ];
      
      case 'server':
        return [
          {
            icon: <CloudOff />,
            title: 'Wait a few minutes',
            description: 'Server issues are usually resolved quickly',
          },
          {
            icon: <Refresh />,
            title: 'Try your request again',
            description: 'The server may be back online',
          },
          ...commonSteps,
        ];
      
      case 'validation':
        return [
          {
            icon: <CheckCircle />,
            title: 'Check your input',
            description: 'Ensure all required fields are filled correctly',
          },
          {
            icon: <Warning />,
            title: 'Review file requirements',
            description: 'Make sure files meet size and format requirements',
          },
          ...commonSteps,
        ];
      
      case 'upload':
        return [
          {
            icon: <Storage />,
            title: 'Check file size and format',
            description: 'Ensure your file is a PDF under 50MB',
          },
          {
            icon: <Wifi />,
            title: 'Check your connection',
            description: 'Large files require stable internet connections',
          },
          {
            icon: <Refresh />,
            title: 'Try uploading again',
            description: 'Sometimes uploads fail due to temporary issues',
          },
          ...commonSteps,
        ];
      
      case 'query':
        return [
          {
            icon: <CheckCircle />,
            title: 'Simplify your query',
            description: 'Try asking a more specific or shorter question',
          },
          {
            icon: <Storage />,
            title: 'Check if documents are uploaded',
            description: 'Ensure you have uploaded documents to query against',
          },
          ...commonSteps,
        ];
      
      default:
        return commonSteps;
    }
  };

  const errorInfo = getErrorTypeInfo(errorType);
  const recoverySteps = getRecoverySteps(errorType);

  const troubleshootingTips = [
    {
      category: 'Browser Issues',
      tips: [
        'Clear your browser cache and cookies',
        'Try using an incognito/private browsing window',
        'Disable browser extensions temporarily',
        'Try a different browser (Chrome, Firefox, Safari)',
      ],
    },
    {
      category: 'File Upload Issues',
      tips: [
        'Ensure file is in PDF format',
        'Check file size is under 50MB',
        'Try uploading from a different location',
        'Ensure filename contains only standard characters',
      ],
    },
    {
      category: 'Query Issues',
      tips: [
        'Make sure you have uploaded documents first',
        'Try shorter, more specific questions',
        'Avoid special characters in queries',
        'Wait for previous queries to complete',
      ],
    },
  ];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      scroll="paper"
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {errorInfo.icon}
          <Box>
            <Typography variant="h6">{errorInfo.title}</Typography>
            <Typography variant="body2" color="text.secondary">
              {errorInfo.description}
            </Typography>
          </Box>
          <Chip
            label={errorType.toUpperCase()}
            color={errorInfo.color}
            size="small"
          />
        </Box>
      </DialogTitle>

      <DialogContent>
        {errorMessage && (
          <Alert severity={errorInfo.color} sx={{ mb: 3 }}>
            <Typography variant="body2">{errorMessage}</Typography>
          </Alert>
        )}

        <Typography variant="h6" gutterBottom>
          Recovery Steps
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Try these steps in order to resolve the issue:
        </Typography>

        <List>
          {recoverySteps.map((step, index) => (
            <ListItem
              key={index}
              sx={{
                cursor: step.action ? 'pointer' : 'default',
                '&:hover': step.action ? { backgroundColor: 'action.hover' } : {},
                borderRadius: 1,
                mb: 1,
              }}
              onClick={step.action}
            >
              <ListItemIcon sx={{ color: 'primary.main' }}>
                {step.icon}
              </ListItemIcon>
              <ListItemText
                primary={step.title}
                secondary={step.description}
              />
            </ListItem>
          ))}
        </List>

        <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
          Additional Troubleshooting
        </Typography>

        {troubleshootingTips.map((section, index) => (
          <Accordion key={index}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle1">{section.category}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List dense>
                {section.tips.map((tip, tipIndex) => (
                  <ListItem key={tipIndex} sx={{ py: 0.5 }}>
                    <ListItemIcon sx={{ minWidth: 32 }}>
                      <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />
                    </ListItemIcon>
                    <ListItemText
                      primary={tip}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        ))}

        <Alert severity="info" sx={{ mt: 3 }}>
          <Typography variant="body2">
            <strong>Still having issues?</strong> Our support team is here to help. 
            Please include the error message and steps you've already tried when contacting us.
          </Typography>
        </Alert>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button
          variant="contained"
          onClick={() => window.open('mailto:support@example.com', '_blank')}
          startIcon={<ContactSupport />}
        >
          Contact Support
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ErrorRecoveryGuide;