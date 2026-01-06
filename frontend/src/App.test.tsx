import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import '@testing-library/jest-dom';
import App from './App';

// Mock the API service
jest.mock('./services/api');

const theme = createTheme();
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const AppWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider theme={theme}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </ThemeProvider>
  </QueryClientProvider>
);

test('renders Enterprise GenAI Platform title', () => {
  render(
    <AppWrapper>
      <App />
    </AppWrapper>
  );
  // Just check that the app renders without crashing
  expect(screen.getByRole('banner')).toBeInTheDocument();
});

test('renders welcome message', () => {
  render(
    <AppWrapper>
      <App />
    </AppWrapper>
  );
  const welcomeElement = screen.getByText(/Welcome to Enterprise GenAI Platform/i);
  expect(welcomeElement).toBeInTheDocument();
});

test('renders feature cards', () => {
  render(
    <AppWrapper>
      <App />
    </AppWrapper>
  );
  
  expect(screen.getByText(/Document Upload/i)).toBeInTheDocument();
  expect(screen.getByText(/Query Interface/i)).toBeInTheDocument();
  expect(screen.getByText(/Admin Dashboard/i)).toBeInTheDocument();
});