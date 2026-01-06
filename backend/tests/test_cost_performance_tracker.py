"""
Tests for Cost and Performance Tracking System

This module tests the comprehensive cost and performance tracking functionality
including token usage counting, cost calculation, budget tracking, and performance metrics.
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from hypothesis import given, strategies as st, assume
from hypothesis import settings

from app.services.cost_performance_tracker import (
    CostPerformanceTracker,
    TokenUsage,
    CostCalculation,
    PerformanceMetrics,
    BudgetAlert,
    ModelProvider
)

class TestCostPerformanceTracker:
    """Test suite for CostPerformanceTracker"""
    
    @pytest.fixture
    def tracker(self):
        """Create a tracker instance for testing"""
        return CostPerformanceTracker(
            daily_budget=10.0,
            monthly_budget=300.0
        )
    
    @pytest.fixture
    def tracker_no_budget(self):
        """Create a tracker instance without budget limits"""
        return CostPerformanceTracker()
    
    def test_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker.daily_budget == 10.0
        assert tracker.monthly_budget == 300.0
        assert len(tracker.budget_alerts) == 4  # 2 daily + 2 monthly alerts
        assert len(tracker.token_usage_history) == 0
        assert len(tracker.performance_metrics) == 0
    
    def test_model_provider_detection(self, tracker):
        """Test model provider detection"""
        assert tracker._get_model_provider("deepseek-chat") == ModelProvider.DEEPSEEK
        assert tracker._get_model_provider("gpt-3.5-turbo") == ModelProvider.OPENAI
        assert tracker._get_model_provider("claude-3") == ModelProvider.ANTHROPIC
        assert tracker._get_model_provider("custom-model") == ModelProvider.CUSTOM
    
    def test_pricing_retrieval(self, tracker):
        """Test pricing information retrieval"""
        deepseek_pricing = tracker._get_pricing("deepseek-chat")
        assert "input_price_per_1k" in deepseek_pricing
        assert "output_price_per_1k" in deepseek_pricing
        assert deepseek_pricing["input_price_per_1k"] == 0.00014
    
    def test_custom_pricing(self):
        """Test custom pricing configuration"""
        custom_pricing = {
            "my-model": {
                "input_price_per_1k": 0.001,
                "output_price_per_1k": 0.002
            }
        }
        tracker = CostPerformanceTracker(custom_pricing=custom_pricing)
        pricing = tracker._get_pricing("my-model")
        assert pricing["input_price_per_1k"] == 0.001
        assert pricing["output_price_per_1k"] == 0.002
    
    def test_token_usage_tracking(self, tracker):
        """Test token usage tracking"""
        usage = tracker.track_token_usage(
            input_tokens=100,
            output_tokens=50,
            model="deepseek-chat",
            operation_type="query"
        )
        
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "deepseek-chat"
        assert usage.operation_type == "query"
        assert len(tracker.token_usage_history) == 1
    
    def test_cost_calculation(self, tracker):
        """Test cost calculation"""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            model="deepseek-chat",
            timestamp=time.time(),
            operation_type="query"
        )
        
        cost = tracker.calculate_cost(usage)
        
        assert isinstance(cost, CostCalculation)
        # DeepSeek pricing: $0.14 per 1M input tokens, $0.28 per 1M output tokens
        expected_input_cost = (1000 / 1000.0) * 0.00014  # $0.00014
        expected_output_cost = (500 / 1000.0) * 0.00028  # $0.00014
        expected_total = expected_input_cost + expected_output_cost
        
        assert abs(cost.input_cost - expected_input_cost) < 1e-8
        assert abs(cost.output_cost - expected_output_cost) < 1e-8
        assert abs(cost.total_cost - expected_total) < 1e-8
        assert len(tracker.cost_history) == 1
    
    def test_performance_tracking(self, tracker):
        """Test performance metrics tracking"""
        operation_id = "test-op-123"
        
        # Start operation
        returned_id = tracker.start_operation(operation_id, "query")
        assert returned_id == operation_id
        assert operation_id in tracker.active_operations
        
        # Simulate some work
        time.sleep(0.01)  # 10ms
        
        # End operation
        metrics = tracker.end_operation(
            operation_id,
            "query",
            success=True,
            metadata={"test": "data"}
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_id == operation_id
        assert metrics.operation_type == "query"
        assert metrics.success is True
        assert metrics.duration_ms >= 10.0  # At least 10ms
        assert metrics.metadata["test"] == "data"
        assert operation_id not in tracker.active_operations
        assert len(tracker.performance_metrics) == 1
    
    def test_failed_operation_tracking(self, tracker):
        """Test tracking of failed operations"""
        operation_id = "failed-op"
        
        tracker.start_operation(operation_id, "query")
        metrics = tracker.end_operation(
            operation_id,
            "query",
            success=False,
            error_message="Test error"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Test error"
    
    def test_daily_cost_tracking(self, tracker):
        """Test daily cost accumulation"""
        # Add some token usage
        usage1 = tracker.track_token_usage(1000, 500, "deepseek-chat")
        usage2 = tracker.track_token_usage(2000, 1000, "deepseek-chat")
        
        # Calculate costs
        cost1 = tracker.calculate_cost(usage1)
        cost2 = tracker.calculate_cost(usage2)
        
        daily_cost = tracker.get_daily_cost()
        expected_cost = cost1.total_cost + cost2.total_cost
        
        assert abs(daily_cost - expected_cost) < 1e-8
    
    def test_monthly_cost_tracking(self, tracker):
        """Test monthly cost accumulation"""
        # Add token usage for today
        usage = tracker.track_token_usage(1000, 500, "deepseek-chat")
        cost = tracker.calculate_cost(usage)
        
        monthly_cost = tracker.get_monthly_cost()
        assert abs(monthly_cost - cost.total_cost) < 1e-8
    
    def test_budget_alerts(self, tracker):
        """Test budget alert triggering"""
        # Initially no alerts should be triggered
        assert all(not alert.triggered for alert in tracker.budget_alerts)
        
        # Add enough usage to trigger 80% daily budget alert
        # Daily budget is $10, so 80% is $8
        # From the test output, 40K tokens cost $0.008400
        # So to reach $8, we need approximately 8 / 0.008400 * 40000 ≈ 38M tokens
        tokens_needed = 40_000_000  # 40 million tokens
        
        usage = tracker.track_token_usage(
            input_tokens=tokens_needed // 2,
            output_tokens=tokens_needed // 2,
            model="deepseek-chat"
        )
        cost = tracker.calculate_cost(usage)
        
        print(f"Generated cost: ${cost.total_cost:.6f} for {tokens_needed:,} tokens")
        print(f"Daily cost: ${tracker.get_daily_cost():.6f}")
        
        # Check if warning alert was triggered
        warning_alerts = [a for a in tracker.budget_alerts if a.alert_type == "warning" and a.threshold_amount <= 10.0]
        assert any(alert.triggered for alert in warning_alerts), f"No daily warning alerts triggered. Alerts: {[(a.alert_type, a.threshold_amount, a.triggered) for a in tracker.budget_alerts]}"
    
    def test_performance_summary(self, tracker):
        """Test performance summary generation"""
        # Add some operations
        for i in range(5):
            op_id = f"op-{i}"
            tracker.start_operation(op_id, "query")
            time.sleep(0.001)  # 1ms
            tracker.end_operation(op_id, "query", success=i < 4)  # 1 failure
        
        summary = tracker.get_performance_summary(hours=1)
        
        assert summary["total_operations"] == 5
        assert summary["success_rate"] == 80.0  # 4/5 * 100
        assert summary["error_count"] == 1
        assert summary["avg_latency_ms"] >= 1.0
        assert summary["throughput_per_hour"] == 5.0
    
    def test_cost_summary(self, tracker):
        """Test cost summary generation"""
        # Add some token usage
        usage = tracker.track_token_usage(1000, 500, "deepseek-chat")
        cost = tracker.calculate_cost(usage)
        
        summary = tracker.get_cost_summary(days=1)
        
        assert summary["total_cost"] == cost.total_cost
        assert summary["total_tokens"] == 1500
        assert summary["avg_daily_cost"] == cost.total_cost
        assert len(summary["daily_breakdown"]) == 1
        assert summary["budget_status"]["daily_budget"] == 10.0
        assert summary["budget_status"]["monthly_budget"] == 300.0
    
    def test_metrics_export(self, tracker):
        """Test metrics export functionality"""
        # Add some data
        usage = tracker.track_token_usage(100, 50, "deepseek-chat")
        tracker.calculate_cost(usage)
        
        op_id = "test-op"
        tracker.start_operation(op_id, "query")
        tracker.end_operation(op_id, "query", success=True)
        
        # Export as JSON
        export_data = tracker.export_metrics("json")
        assert isinstance(export_data, str)
        
        import json
        data = json.loads(export_data)
        assert "token_usage" in data
        assert "performance_metrics" in data
        assert "daily_stats" in data
        assert len(data["token_usage"]) == 1
        assert len(data["performance_metrics"]) == 1
    
    def test_metrics_reset(self, tracker):
        """Test metrics reset functionality"""
        # Add some data
        tracker.track_token_usage(100, 50, "deepseek-chat")
        tracker.start_operation("test", "query")
        
        assert len(tracker.token_usage_history) > 0
        assert len(tracker.active_operations) > 0
        
        # Reset should require confirmation
        with pytest.raises(ValueError):
            tracker.reset_metrics()
        
        # Reset with confirmation
        tracker.reset_metrics(confirm=True)
        
        assert len(tracker.token_usage_history) == 0
        assert len(tracker.cost_history) == 0
        assert len(tracker.performance_metrics) == 0
        assert len(tracker.active_operations) == 0
        assert len(tracker.daily_stats) == 0
    
    def test_thread_safety(self, tracker):
        """Test thread safety of tracker operations"""
        import threading
        import concurrent.futures
        
        def add_usage(thread_id):
            for i in range(10):
                usage = tracker.track_token_usage(
                    input_tokens=100,
                    output_tokens=50,
                    model="deepseek-chat",
                    operation_type=f"thread-{thread_id}"
                )
                tracker.calculate_cost(usage)
        
        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_usage, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        # Should have 50 total usage records (5 threads * 10 operations)
        assert len(tracker.token_usage_history) == 50
        assert len(tracker.cost_history) == 50
    
    def test_no_budget_initialization(self, tracker_no_budget):
        """Test initialization without budget limits"""
        assert tracker_no_budget.daily_budget is None
        assert tracker_no_budget.monthly_budget is None
        assert len(tracker_no_budget.budget_alerts) == 0
    
    def test_invalid_export_format(self, tracker):
        """Test invalid export format handling"""
        with pytest.raises(ValueError, match="Unsupported export format"):
            tracker.export_metrics("xml")


class TestCostTrackingProperties:
    """Property-based tests for cost tracking accuracy"""
    
    @given(
        input_tokens=st.integers(min_value=1, max_value=1_000_000),
        output_tokens=st.integers(min_value=1, max_value=1_000_000),
        model_name=st.sampled_from([
            "deepseek-chat", "deepseek-coder", 
            "gpt-3.5-turbo", "gpt-4",
            "claude-3-sonnet", "claude-3-haiku",
            "custom-model-1", "unknown-model"
        ]),
        operation_type=st.sampled_from(["query", "embedding", "rerank", "summarization"])
    )
    @settings(max_examples=100)
    def test_property_16_cost_tracking_accuracy(self, input_tokens, output_tokens, model_name, operation_type):
        """
        **Feature: enterprise-genai-platform, Property 16: Cost Tracking Accuracy**
        
        For any API call, the system should log accurate token usage counts 
        and calculate associated costs correctly.
        
        **Validates: Requirements 5.3**
        """
        # Create tracker instance
        tracker = CostPerformanceTracker()
        
        # Track token usage (simulating an API call)
        usage = tracker.track_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
            operation_type=operation_type
        )
        
        # Verify token usage is logged accurately
        assert len(tracker.token_usage_history) == 1
        logged_usage = tracker.token_usage_history[0]
        
        # Property: Token counts must be logged exactly as provided
        assert logged_usage.input_tokens == input_tokens
        assert logged_usage.output_tokens == output_tokens
        assert logged_usage.total_tokens == input_tokens + output_tokens
        assert logged_usage.model == model_name
        assert logged_usage.operation_type == operation_type
        
        # Calculate cost for the usage
        cost = tracker.calculate_cost(usage)
        
        # Verify cost calculation accuracy
        assert len(tracker.cost_history) == 1
        logged_cost = tracker.cost_history[0]
        
        # Property: Cost calculation must be mathematically correct
        pricing = tracker._get_pricing(model_name)
        expected_input_cost = (input_tokens / 1000.0) * pricing["input_price_per_1k"]
        expected_output_cost = (output_tokens / 1000.0) * pricing["output_price_per_1k"]
        expected_total_cost = expected_input_cost + expected_output_cost
        
        # Allow for floating point precision differences
        assert abs(cost.input_cost - expected_input_cost) < 1e-10
        assert abs(cost.output_cost - expected_output_cost) < 1e-10
        assert abs(cost.total_cost - expected_total_cost) < 1e-10
        assert abs(logged_cost.total_cost - expected_total_cost) < 1e-10
        
        # Property: Daily cost tracking must accumulate correctly
        daily_cost_before = tracker.get_daily_cost()
        
        # Add another API call
        usage2 = tracker.track_token_usage(
            input_tokens=input_tokens // 2,
            output_tokens=output_tokens // 2,
            model=model_name,
            operation_type=operation_type
        )
        cost2 = tracker.calculate_cost(usage2)
        
        daily_cost_after = tracker.get_daily_cost()
        
        # Property: Daily cost should increase by exactly the new cost
        expected_daily_increase = cost2.total_cost
        actual_daily_increase = daily_cost_after - daily_cost_before
        assert abs(actual_daily_increase - expected_daily_increase) < 1e-10
        
        # Property: Total daily cost should equal sum of all individual costs
        expected_total_daily_cost = cost.total_cost + cost2.total_cost
        assert abs(daily_cost_after - expected_total_daily_cost) < 1e-10


class TestTokenUsage:
    """Test TokenUsage dataclass"""
    
    def test_token_usage_creation(self):
        """Test TokenUsage object creation"""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="test-model",
            timestamp=time.time(),
            operation_type="test"
        )
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "test-model"
        assert usage.operation_type == "test"
    
    def test_token_usage_auto_total(self):
        """Test automatic total calculation"""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=0,  # Will be recalculated
            model="test-model",
            timestamp=time.time(),
            operation_type="test"
        )
        
        assert usage.total_tokens == 150  # 100 + 50

class TestCostCalculation:
    """Test CostCalculation dataclass"""
    
    def test_cost_calculation_creation(self):
        """Test CostCalculation object creation"""
        cost = CostCalculation(
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            currency="USD"
        )
        
        assert cost.input_cost == 0.001
        assert cost.output_cost == 0.002
        assert cost.total_cost == 0.003
        assert cost.currency == "USD"

class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics object creation"""
        start_time = time.time()
        end_time = start_time + 0.1
        
        metrics = PerformanceMetrics(
            operation_id="test-123",
            operation_type="query",
            start_time=start_time,
            end_time=end_time,
            duration_ms=100.0,
            success=True,
            metadata={"key": "value"}
        )
        
        assert metrics.operation_id == "test-123"
        assert metrics.operation_type == "query"
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.duration_ms == 100.0
        assert metrics.success is True
        assert metrics.metadata["key"] == "value"

class TestBudgetAlert:
    """Test BudgetAlert dataclass"""
    
    def test_budget_alert_creation(self):
        """Test BudgetAlert object creation"""
        alert = BudgetAlert(
            threshold_percent=80.0,
            threshold_amount=8.0,
            alert_type="warning"
        )
        
        assert alert.threshold_percent == 80.0
        assert alert.threshold_amount == 8.0
        assert alert.alert_type == "warning"
        assert alert.triggered is False
        assert alert.last_triggered is None