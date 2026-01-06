# Makefile for Enterprise GenAI Platform

.PHONY: help build up down logs clean test lint format install-backend install-frontend

# Default target
help:
	@echo "Available commands:"
	@echo "  build          - Build all Docker images"
	@echo "  up             - Start all services in production mode"
	@echo "  up-dev         - Start all services in development mode"
	@echo "  down           - Stop all services"
	@echo "  logs           - Show logs from all services"
	@echo "  logs-backend   - Show backend logs"
	@echo "  logs-frontend  - Show frontend logs"
	@echo "  clean          - Remove all containers and volumes"
	@echo "  test           - Run all tests"
	@echo "  lint           - Run linting for all code"
	@echo "  format         - Format all code"
	@echo "  install-backend - Install backend dependencies locally"
	@echo "  install-frontend - Install frontend dependencies locally"
	@echo "  health         - Check service health"
	@echo "  test-deepseek  - Test DeepSeek API connection"
	@echo "  setup-deepseek - Show DeepSeek setup instructions"

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

up-dev:
	docker-compose -f docker-compose.dev.yml up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# Development commands
install-backend:
	cd backend && pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

test:
	@echo "Running backend tests..."
	cd backend && python -m pytest
	@echo "Running frontend tests..."
	cd frontend && npm test -- --run

lint:
	@echo "Linting backend..."
	cd backend && flake8 app/
	cd backend && mypy app/
	@echo "Linting frontend..."
	cd frontend && npm run lint

format:
	@echo "Formatting backend..."
	cd backend && black app/
	@echo "Formatting frontend..."
	cd frontend && npm run format

# Health check
health:
	@echo "Checking backend health..."
	curl -f http://localhost:8000/health || echo "Backend not responding"
	@echo "Checking frontend health..."
	curl -f http://localhost:3000/ || echo "Frontend not responding"

# DeepSeek testing
test-deepseek:
	@echo "Testing DeepSeek API connection..."
	python test_deepseek.py

# Setup with DeepSeek
setup-deepseek:
	@echo "Setting up DeepSeek configuration..."
	@echo "Please ensure you have set DEEPSEEK_API_KEY environment variable"
	@echo "Get your API key from: https://platform.deepseek.com/"
	@echo "Then run: export DEEPSEEK_API_KEY=your-actual-api-key"