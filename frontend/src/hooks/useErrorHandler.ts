import { useCallback } from 'react';
import { useNotification } from '../components/NotificationSystem';

export interface ErrorHandlerOptions {
  showNotification?: boolean;
  title?: string;
  fallbackMessage?: string;
  logError?: boolean;
}

export const useErrorHandler = () => {
  const { showError, showWarning } = useNotification();

  const handleError = useCallback((
    error: Error | string | unknown,
    options: ErrorHandlerOptions = {}
  ) => {
    const {
      showNotification = true,
      title = 'Error',
      fallbackMessage = 'An unexpected error occurred',
      logError = true,
    } = options;

    let errorMessage: string;
    let errorObject: Error | null = null;

    // Extract error message
    if (error instanceof Error) {
      errorObject = error;
      errorMessage = error.message;
    } else if (typeof error === 'string') {
      errorMessage = error;
    } else if (error && typeof error === 'object' && 'message' in error) {
      errorMessage = (error as any).message;
    } else {
      errorMessage = fallbackMessage;
    }

    // Log error for debugging
    if (logError) {
      console.error('Error handled:', errorObject || error);
    }

    // Show notification
    if (showNotification) {
      showError(errorMessage, title);
    }

    return {
      message: errorMessage,
      error: errorObject,
    };
  }, [showError]);

  const handleApiError = useCallback((
    error: any,
    options: ErrorHandlerOptions = {}
  ) => {
    let errorMessage = options.fallbackMessage || 'Failed to communicate with server';
    let title = options.title || 'Network Error';

    // Handle different types of API errors
    if (error?.response) {
      // Server responded with error status
      const status = error.response.status;
      const data = error.response.data;

      switch (status) {
        case 400:
          title = 'Bad Request';
          errorMessage = data?.message || 'Invalid request. Please check your input.';
          break;
        case 401:
          title = 'Unauthorized';
          errorMessage = 'You are not authorized to perform this action.';
          break;
        case 403:
          title = 'Forbidden';
          errorMessage = 'Access denied. You do not have permission to access this resource.';
          break;
        case 404:
          title = 'Not Found';
          errorMessage = 'The requested resource was not found.';
          break;
        case 429:
          title = 'Rate Limited';
          errorMessage = 'Too many requests. Please wait a moment and try again.';
          break;
        case 500:
          title = 'Server Error';
          errorMessage = 'Internal server error. Please try again later.';
          break;
        case 502:
        case 503:
        case 504:
          title = 'Service Unavailable';
          errorMessage = 'Service is temporarily unavailable. Please try again later.';
          break;
        default:
          errorMessage = data?.message || `Server error (${status})`;
      }
    } else if (error?.request) {
      // Network error
      title = 'Connection Error';
      errorMessage = 'Unable to connect to server. Please check your internet connection.';
    } else if (error?.message) {
      // Other error
      errorMessage = error.message;
    }

    return handleError(errorMessage, { ...options, title });
  }, [handleError]);

  const handleValidationError = useCallback((
    validationErrors: Record<string, string[]> | string[],
    options: ErrorHandlerOptions = {}
  ) => {
    let errorMessage: string;

    if (Array.isArray(validationErrors)) {
      errorMessage = validationErrors.join(', ');
    } else {
      const errors = Object.values(validationErrors).flat();
      errorMessage = errors.join(', ');
    }

    return handleError(errorMessage, {
      ...options,
      title: options.title || 'Validation Error',
    });
  }, [handleError]);

  const handleWarning = useCallback((
    message: string,
    title: string = 'Warning'
  ) => {
    showWarning(message, title);
  }, [showWarning]);

  return {
    handleError,
    handleApiError,
    handleValidationError,
    handleWarning,
  };
};

export default useErrorHandler;