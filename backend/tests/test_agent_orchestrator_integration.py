"""
Integration tests for the Agent Orchestration System

Tests the integration of the agent orchestrator with the services package.
"""

import pytest
from app.services import (
    AgentOrchestrator,
    AgentState,
    create_agent_orchestrator
)


class TestAgentOrchestratorIntegration:
    """Test integration of agent orchestrator with services package"""
    
    def test_import_from_services_package(self):
        """Test that agent orchestrator can be imported from services package"""
        # This test passes if the imports above work
        assert AgentOrchestrator is not None
        assert AgentState is not None
        assert create_agent_orchestrator is not None
    
    def test_create_orchestrator_from_services(self):
        """Test creating orchestrator using the factory function from services"""
        orchestrator = create_agent_orchestrator()
        
        assert isinstance(orchestrator, AgentOrchestrator)
        assert orchestrator.workflow is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_integration(self):
        """Test end-to-end workflow processing"""
        orchestrator = create_agent_orchestrator()
        query = "Test integration query"
        
        result = await orchestrator.process_query(query)
        
        # Verify the workflow completed successfully
        assert result.query == query
        assert result.plan is not None
        assert result.response is not None
        assert result.evaluation_score is not None
        assert result.iteration_count > 0
        assert result.processing_start_time is not None
        assert result.error_message is None