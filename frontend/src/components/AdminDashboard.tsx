import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Alert,
  Button,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  Dashboard,
  Storage,
  Speed,
  AttachMoney,
  People,
  Description,
  CheckCircle,
  Error,
  Warning,
  Refresh,
  Settings,
  TrendingUp,
  Memory,
  Timer,
  CloudQueue,
} from '@mui/icons-material';
import { apiService } from '../services/api';
import { SystemHealth } from '../types';

interface SystemMetrics {
  total_documents: number;
  total_queries: number;
  avg_response_time: number;
  total_tokens_used: number;
  estimated_cost: number;
  cache_hit_rate: number;
  system_uptime: number;
  active_users: number;
}

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime: number;
  last_check: string;
  response_time: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
  </div>
);

const AdminDashboard: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchDashboardData = async () => {
    try {
      setError(null);
      
      // Fetch system health
      const health = await apiService.getHealth();
      setSystemHealth(health);
      
      // Fetch metrics (mock data for now)
      const mockMetrics: SystemMetrics = {
        total_documents: 156,
        total_queries: 2847,
        avg_response_time: 1.2,
        total_tokens_used: 1250000,
        estimated_cost: 125.50,
        cache_hit_rate: 0.78,
        system_uptime: health.uptime,
        active_users: 23,
      };
      setMetrics(mockMetrics);
      
      // Mock service status data
      const mockServices: ServiceStatus[] = [
        {
          name: 'API Gateway',
          status: health.services.api ? 'healthy' : 'unhealthy',
          uptime: 99.9,
          last_check: new Date().toISOString(),
          response_time: 45,
        },
        {
          name: 'Vector Database',
          status: health.services.vector_db ? 'healthy' : 'unhealthy',
          uptime: 99.8,
          last_check: new Date().toISOString(),
          response_time: 120,
        },
        {
          name: 'Document Store',
          status: health.services.document_store ? 'healthy' : 'unhealthy',
          uptime: 99.95,
          last_check: new Date().toISOString(),
          response_time: 30,
        },
        {
          name: 'Embedding Service',
          status: health.services.embedding_service ? 'healthy' : 'degraded',
          uptime: 98.5,
          last_check: new Date().toISOString(),
          response_time: 850,
        },
      ];
      setServices(mockServices);
      
    } catch (err: any) {
      setError(err.message || 'Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const getStatusColor = (status: string): 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'primary';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" />;
      case 'degraded': return <Warning color="warning" />;
      case 'unhealthy': return <Error color="error" />;
      default: return <CheckCircle />;
    }
  };

  const formatUptime = (seconds: number): string => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  };

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <LinearProgress sx={{ width: 300 }} />
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Admin Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={fetchDashboardData}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)} sx={{ mb: 3 }}>
        <Tab label="Overview" />
        <Tab label="Performance" />
        <Tab label="Services" />
        <Tab label="Users" />
      </Tabs>

      {/* Overview Tab */}
      <TabPanel value={tabValue} index={0}>
        {/* System Status Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Description color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Documents</Typography>
                </Box>
                <Typography variant="h4" color="primary">
                  {metrics?.total_documents || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total processed
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <CloudQueue color="success" sx={{ mr: 1 }} />
                  <Typography variant="h6">Queries</Typography>
                </Box>
                <Typography variant="h4" color="success.main">
                  {metrics?.total_queries || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total processed
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Timer color="info" sx={{ mr: 1 }} />
                  <Typography variant="h6">Avg Response</Typography>
                </Box>
                <Typography variant="h4" color="info.main">
                  {metrics?.avg_response_time || 0}s
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Response time
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <AttachMoney color="warning" sx={{ mr: 1 }} />
                  <Typography variant="h6">Cost</Typography>
                </Box>
                <Typography variant="h4" color="warning.main">
                  {formatCurrency(metrics?.estimated_cost || 0)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This month
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* System Health */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Health
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <Chip
                icon={getStatusIcon(systemHealth?.status || 'healthy')}
                label={systemHealth?.status?.toUpperCase() || 'UNKNOWN'}
                color={getStatusColor(systemHealth?.status || 'healthy')}
              />
              <Typography variant="body2" color="text.secondary">
                Uptime: {formatUptime(systemHealth?.uptime || 0)}
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={systemHealth?.status === 'healthy' ? 100 : systemHealth?.status === 'degraded' ? 75 : 25}
              color={getStatusColor(systemHealth?.status || 'healthy')}
            />
          </CardContent>
        </Card>
      </TabPanel>

      {/* Performance Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Cache Performance
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Memory color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h4" color="primary">
                    {Math.round((metrics?.cache_hit_rate || 0) * 100)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(metrics?.cache_hit_rate || 0) * 100}
                  sx={{ mb: 1 }}
                />
                <Typography variant="body2" color="text.secondary">
                  Cache hit rate
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Token Usage
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TrendingUp color="success" sx={{ mr: 1 }} />
                  <Typography variant="h4" color="success.main">
                    {(metrics?.total_tokens_used || 0).toLocaleString()}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Total tokens processed
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Services Tab */}
      <TabPanel value={tabValue} index={2}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Service</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Uptime</TableCell>
                <TableCell>Response Time</TableCell>
                <TableCell>Last Check</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {services.map((service) => (
                <TableRow key={service.name}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {getStatusIcon(service.status)}
                      <Typography sx={{ ml: 1 }}>{service.name}</Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={service.status}
                      color={getStatusColor(service.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{service.uptime}%</TableCell>
                  <TableCell>{service.response_time}ms</TableCell>
                  <TableCell>
                    {new Date(service.last_check).toLocaleTimeString()}
                  </TableCell>
                  <TableCell>
                    <Tooltip title="Restart Service">
                      <IconButton size="small">
                        <Refresh />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Configure">
                      <IconButton size="small">
                        <Settings />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Users Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Active Users
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <People color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h4" color="primary">
                    {metrics?.active_users || 0}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Currently online
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  User Management
                </Typography>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <People />
                    </ListItemIcon>
                    <ListItemText
                      primary="User Roles & Permissions"
                      secondary="Manage user access levels and permissions"
                    />
                    <Button variant="outlined" size="small">
                      Manage
                    </Button>
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemIcon>
                      <Settings />
                    </ListItemIcon>
                    <ListItemText
                      primary="System Configuration"
                      secondary="Configure system settings and parameters"
                    />
                    <Button variant="outlined" size="small">
                      Configure
                    </Button>
                  </ListItem>
                  <Divider />
                  <ListItem>
                    <ListItemIcon>
                      <Storage />
                    </ListItemIcon>
                    <ListItemText
                      primary="Data Management"
                      secondary="Manage documents and vector database"
                    />
                    <Button variant="outlined" size="small">
                      Manage
                    </Button>
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  );
};

export default AdminDashboard;