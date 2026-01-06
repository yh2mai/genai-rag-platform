# Enterprise GenAI Knowledge Agent Platform

A production-grade system that transforms enterprise PDF documents into an intelligent knowledge base using advanced RAG (Retrieval-Augmented Generation) pipelines and agentic workflows.

## Architecture Overview

The platform employs a microservices architecture with:
- **Backend**: FastAPI-based Python service with RAG pipeline and agent orchestration
- **Frontend**: React TypeScript application with Material-UI
- **Vector Database**: FAISS for semantic search
- **Document Storage**: Local filesystem with metadata indexing
- **Caching**: Redis for response caching and performance optimization

## Features

- **Document Processing**: Intelligent PDF ingestion with semantic chunking
- **Hybrid Retrieval**: Combines vector similarity and keyword search
- **Agent Orchestration**: Multi-agent workflows using LangGraph
- **LLMOps Monitoring**: Comprehensive tracking and governance
- **Evaluation Framework**: Automated quality assessment
- **Enterprise Ready**: Docker containerization and cloud deployment support

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local backend development)
- DeepSeek API key (get one at https://platform.deepseek.com/)

### Using Docker Compose (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd enterprise-genai-platform
   ```

2. **Start all services**:
   ```bash
   # Production mode
   docker-compose up -d

   # Development mode with hot reload
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development

#### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration, especially:
   # - DEEPSEEK_API_KEY=your-actual-api-key
   ```

5. **Run the server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm start
   ```

## Project Structure

```
enterprise-genai-platform/
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── core/              # Configuration and logging
│   │   ├── main.py            # FastAPI application entry point
│   │   └── __init__.py
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Backend container configuration
│   └── .env.example          # Environment variables template
├── frontend/                  # React TypeScript frontend
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API services
│   │   ├── types/           # TypeScript type definitions
│   │   ├── App.tsx          # Main application component
│   │   └── index.tsx        # Application entry point
│   ├── package.json         # Node.js dependencies
│   ├── Dockerfile          # Frontend container configuration
│   └── nginx.conf          # Nginx configuration
├── docker-compose.yml       # Production Docker Compose
├── docker-compose.dev.yml   # Development Docker Compose
└── README.md               # This file
```

## Configuration

### DeepSeek API Setup

The platform uses DeepSeek as the primary language model. To configure:

1. **Get API Key**: Sign up at https://platform.deepseek.com/ and obtain your API key
2. **Set Environment Variable**: Add your API key to the `.env` file:
   ```bash
   DEEPSEEK_API_KEY=your-actual-deepseek-api-key-here
   ```
3. **Docker Environment**: For Docker deployments, set the environment variable:
   ```bash
   export DEEPSEEK_API_KEY=your-actual-deepseek-api-key-here
   make up
   ```

### Environment Variables

The backend uses the following key environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Enable debug mode (default: false)
- `DATABASE_URL`: Database connection string
- `VECTOR_DB_PATH`: Vector database storage path
- `DOCUMENT_STORE_PATH`: Document storage path
- `EMBEDDING_MODEL`: HuggingFace embedding model
- `LLM_MODEL`: DeepSeek language model (default: deepseek-chat)
- `LLM_BASE_URL`: DeepSeek API base URL
- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `MAX_TOKENS`: Maximum tokens for LLM responses
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `REDIS_URL`: Redis connection string

### Docker Configuration

The application supports multiple deployment modes:

- **Production**: `docker-compose.yml` - Optimized builds with health checks
- **Development**: `docker-compose.dev.yml` - Hot reload and debug mode

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Health Checks

The system includes comprehensive health monitoring:
- Backend health: `GET /health`
- System metrics: `GET /metrics` (when implemented)
- Container health checks via Docker

## Development Guidelines

### Code Quality

- **Backend**: Uses Black for formatting, Flake8 for linting, MyPy for type checking
- **Frontend**: Uses Prettier for formatting, ESLint for linting
- **Testing**: Pytest for backend, Jest/React Testing Library for frontend

### Logging

The application uses structured logging with:
- Console output for development
- File rotation for production
- Configurable log levels
- Request/response logging

## Deployment

### Docker Production Deployment

1. **Build and deploy**:
   ```bash
   docker-compose up -d --build
   ```

2. **Monitor logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Scale services**:
   ```bash
   docker-compose up -d --scale backend=3
   ```

### Cloud Deployment

The application is designed for cloud deployment with:
- Container orchestration support (Kubernetes manifests coming in later tasks)
- Environment-based configuration
- Health check endpoints
- Horizontal scaling capabilities

## Monitoring and Observability

- **Health Checks**: Built-in health endpoints
- **Logging**: Structured logging with rotation
- **Metrics**: Performance and usage tracking (implementation in later tasks)
- **Error Tracking**: Comprehensive error handling and reporting

## Security Considerations

- **CORS**: Configurable cross-origin resource sharing
- **Environment Variables**: Sensitive data via environment configuration
- **Input Validation**: Request validation using Pydantic
- **Security Headers**: Nginx security headers in production

## Contributing

1. Follow the established code style and formatting
2. Write tests for new functionality
3. Update documentation for API changes
4. Use conventional commit messages

## License

[Add your license information here]

## Support

For questions and support:
- Check the API documentation at `/docs`
- Review the application logs
- Consult the troubleshooting section below

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 3000, 8000, and 6379 are available
2. **Docker issues**: Check Docker daemon is running and has sufficient resources
3. **Permission errors**: Ensure proper file permissions for volume mounts
4. **Network connectivity**: Verify Docker network configuration

### Debugging

- Enable debug mode: Set `DEBUG=true` in environment
- Check logs: `docker-compose logs [service-name]`
- Access containers: `docker-compose exec [service-name] /bin/bash`