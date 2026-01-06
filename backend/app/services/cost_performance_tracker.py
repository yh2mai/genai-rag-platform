"""
Cost and Performance Tracking System for LLMOps Monitoring

This module provides comprehensive tracking of token usage, cost calculation,
budget management, and performance metrics for the Enterprise GenAI Platform.
Supports real-time monitoring, alerting, and reporting capabilities.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported LLM providers with their pricing models"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

@dataclass
class TokenUsage:
    """Token usage information for a single API call"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    timestamp: float
    operation_type: str  # e.g., "query", "embedding", "rerank"
    
    def __post_init__(self):
        if self.total_tokens != self.input_tokens + self.output_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens

@dataclass
class CostCalculation:
    """Cost calculation for token usage"""
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    
@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation"""
    operation_id: str
    operation_type: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BudgetAlert:
    """Budget alert configuration and status"""
    threshold_percent: float
    threshold_amount: float
    alert_type: str  # "warning", "critical"
    triggered: bool = False
    last_triggered: Optional[float] = None

class CostPerformanceTracker:
    """
    Comprehensive cost and performance tracking system.
    
    Features:
    - Real-time token usage counting and cost calculation
    - Budget tracking with configurable alerts
    - Performance metrics collection (latency, throughput, error rates)
    - Historical data storage and reporting
    - Multi-model support with different pricing structures
    """
    
    # Default pricing per 1K tokens (USD)
    DEFAULT_PRICING = {
        ModelProvider.DEEPSEEK: {
            "input_price_per_1k": 0.00014,  # $0.14 per 1M tokens
            "output_price_per_1k": 0.00028  # $0.28 per 1M tokens
        },
        ModelProvider.OPENAI: {
            "input_price_per_1k": 0.0015,   # GPT-3.5-turbo pricing
            "output_price_per_1k": 0.002
        },
        ModelProvider.ANTHROPIC: {
            "input_price_per_1k": 0.0008,   # Claude pricing
            "output_price_per_1k": 0.0024
        }
    }
    
    def __init__(self, 
                 daily_budget: Optional[float] = None,
                 monthly_budget: Optional[float] = None,
                 custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the cost and performance tracker.
        
        Args:
            daily_budget: Daily spending limit in USD
            monthly_budget: Monthly spending limit in USD
            custom_pricing: Custom pricing configuration for models
        """
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.custom_pricing = custom_pricing or {}
        
        # Thread-safe storage for metrics
        self._lock = threading.Lock()
        
        # Token usage and cost tracking
        self.token_usage_history: List[TokenUsage] = []
        self.cost_history: List[CostCalculation] = []
        
        # Performance metrics
        self.performance_metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
        
        # Budget alerts
        self.budget_alerts: List[BudgetAlert] = []
        self._setup_default_alerts()
        
        # Aggregated statistics
        self.daily_stats = defaultdict(lambda: {
            "total_tokens": 0,
            "total_cost": 0.0,
            "operation_count": 0,
            "avg_latency_ms": 0.0,
            "error_count": 0
        })
        
        # Performance windows for real-time metrics
        self.latency_window = deque(maxlen=1000)  # Last 1000 operations
        self.throughput_window = deque(maxlen=100)  # Last 100 time windows
        
        logger.info("Cost and Performance Tracker initialized")
    
    def _setup_default_alerts(self):
        """Setup default budget alerts"""
        if self.daily_budget:
            self.budget_alerts.extend([
                BudgetAlert(threshold_percent=80.0, threshold_amount=self.daily_budget * 0.8, alert_type="warning"),
                BudgetAlert(threshold_percent=95.0, threshold_amount=self.daily_budget * 0.95, alert_type="critical")
            ])
        
        if self.monthly_budget:
            self.budget_alerts.extend([
                BudgetAlert(threshold_percent=80.0, threshold_amount=self.monthly_budget * 0.8, alert_type="warning"),
                BudgetAlert(threshold_percent=95.0, threshold_amount=self.monthly_budget * 0.95, alert_type="critical")
            ])
    
    def _get_model_provider(self, model_name: str) -> ModelProvider:
        """Determine the provider based on model name"""
        model_lower = model_name.lower()
        if "deepseek" in model_lower:
            return ModelProvider.DEEPSEEK
        elif "gpt" in model_lower or "openai" in model_lower:
            return ModelProvider.OPENAI
        elif "claude" in model_lower or "anthropic" in model_lower:
            return ModelProvider.ANTHROPIC
        else:
            return ModelProvider.CUSTOM
    
    def _get_pricing(self, model_name: str) -> Dict[str, float]:
        """Get pricing information for a model"""
        provider = self._get_model_provider(model_name)
        
        # Check custom pricing first
        if model_name in self.custom_pricing:
            return self.custom_pricing[model_name]
        
        # Use default pricing
        if provider in self.DEFAULT_PRICING:
            return self.DEFAULT_PRICING[provider]
        
        # Fallback to DeepSeek pricing for unknown models
        logger.warning(f"Unknown model {model_name}, using DeepSeek pricing")
        return self.DEFAULT_PRICING[ModelProvider.DEEPSEEK]
    
    def track_token_usage(self, 
                         input_tokens: int,
                         output_tokens: int,
                         model: str,
                         operation_type: str = "query") -> TokenUsage:
        """
        Track token usage for an API call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name used
            operation_type: Type of operation (query, embedding, etc.)
            
        Returns:
            TokenUsage object with recorded information
        """
        with self._lock:
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                model=model,
                timestamp=time.time(),
                operation_type=operation_type
            )
            
            self.token_usage_history.append(usage)
            
            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_stats[today]["total_tokens"] += usage.total_tokens
            self.daily_stats[today]["operation_count"] += 1
            
            logger.debug(f"Tracked token usage: {usage.total_tokens} tokens for {model}")
            return usage
    
    def calculate_cost(self, token_usage: TokenUsage) -> CostCalculation:
        """
        Calculate cost for token usage.
        
        Args:
            token_usage: TokenUsage object to calculate cost for
            
        Returns:
            CostCalculation with detailed cost breakdown
        """
        pricing = self._get_pricing(token_usage.model)
        
        # Calculate costs (pricing is per 1K tokens)
        input_cost = (token_usage.input_tokens / 1000.0) * pricing["input_price_per_1k"]
        output_cost = (token_usage.output_tokens / 1000.0) * pricing["output_price_per_1k"]
        total_cost = input_cost + output_cost
        
        cost_calc = CostCalculation(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        
        with self._lock:
            self.cost_history.append(cost_calc)
            
            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_stats[today]["total_cost"] += total_cost
            
            # Check budget alerts
            self._check_budget_alerts()
        
        logger.debug(f"Calculated cost: ${total_cost:.6f} for {token_usage.total_tokens} tokens")
        return cost_calc
    
    def start_operation(self, operation_id: str, operation_type: str) -> str:
        """
        Start tracking performance for an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (query, document_processing, etc.)
            
        Returns:
            Operation ID for tracking
        """
        start_time = time.time()
        
        with self._lock:
            self.active_operations[operation_id] = start_time
        
        logger.debug(f"Started tracking operation {operation_id} ({operation_type})")
        return operation_id
    
    def end_operation(self, 
                     operation_id: str,
                     operation_type: str,
                     success: bool = True,
                     error_message: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """
        End tracking for an operation and record performance metrics.
        
        Args:
            operation_id: Operation identifier
            operation_type: Type of operation
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            metadata: Additional metadata about the operation
            
        Returns:
            PerformanceMetrics object with recorded information
        """
        end_time = time.time()
        
        with self._lock:
            start_time = self.active_operations.pop(operation_id, end_time)
            duration_ms = (end_time - start_time) * 1000.0
            
            metrics = PerformanceMetrics(
                operation_id=operation_id,
                operation_type=operation_type,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )
            
            self.performance_metrics.append(metrics)
            
            # Update performance windows
            self.latency_window.append(duration_ms)
            
            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            current_avg = self.daily_stats[today]["avg_latency_ms"]
            current_count = self.daily_stats[today]["operation_count"]
            
            # Calculate new average latency
            if current_count > 0:
                new_avg = ((current_avg * (current_count - 1)) + duration_ms) / current_count
                self.daily_stats[today]["avg_latency_ms"] = new_avg
            else:
                self.daily_stats[today]["avg_latency_ms"] = duration_ms
            
            if not success:
                self.daily_stats[today]["error_count"] += 1
        
        logger.debug(f"Completed operation {operation_id}: {duration_ms:.2f}ms, success={success}")
        return metrics
    
    def _check_budget_alerts(self):
        """Check if any budget thresholds have been exceeded"""
        current_daily_cost = self.get_daily_cost()
        current_monthly_cost = self.get_monthly_cost()
        
        for alert in self.budget_alerts:
            should_trigger = False
            
            # Check daily budget alerts
            if (self.daily_budget and 
                alert.threshold_amount <= self.daily_budget and 
                current_daily_cost >= alert.threshold_amount):
                should_trigger = True
            
            # Check monthly budget alerts  
            elif (self.monthly_budget and 
                  alert.threshold_amount <= self.monthly_budget and 
                  current_monthly_cost >= alert.threshold_amount):
                should_trigger = True
            
            if should_trigger and not alert.triggered:
                alert.triggered = True
                alert.last_triggered = time.time()
                logger.warning(f"Budget alert triggered: {alert.alert_type} at {alert.threshold_percent}%")
    
    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get total cost for a specific day"""
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        return self.daily_stats[target_date]["total_cost"]
    
    def get_monthly_cost(self, year_month: Optional[str] = None) -> float:
        """Get total cost for a specific month"""
        target_month = year_month or datetime.now().strftime("%Y-%m")
        
        total_cost = 0.0
        for date, stats in self.daily_stats.items():
            if date.startswith(target_month):
                total_cost += stats["total_cost"]
        
        return total_cost
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with performance metrics
        """
        cutoff_time = time.time() - (hours * 3600)
        
        recent_metrics = [m for m in self.performance_metrics if m.start_time >= cutoff_time]
        
        if not recent_metrics:
            return {
                "total_operations": 0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0,
                "error_count": 0,
                "throughput_per_hour": 0.0
            }
        
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        avg_latency = sum(m.duration_ms for m in recent_metrics) / total_operations
        error_count = total_operations - successful_operations
        success_rate = (successful_operations / total_operations) * 100.0
        throughput_per_hour = total_operations / hours
        
        return {
            "total_operations": total_operations,
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": round(success_rate, 2),
            "error_count": error_count,
            "throughput_per_hour": round(throughput_per_hour, 2)
        }
    
    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get cost summary for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with cost metrics
        """
        total_cost = 0.0
        total_tokens = 0
        daily_costs = []
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            day_stats = self.daily_stats[date]
            daily_costs.append({
                "date": date,
                "cost": day_stats["total_cost"],
                "tokens": day_stats["total_tokens"]
            })
            total_cost += day_stats["total_cost"]
            total_tokens += day_stats["total_tokens"]
        
        avg_daily_cost = total_cost / days if days > 0 else 0.0
        
        return {
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
            "avg_daily_cost": round(avg_daily_cost, 6),
            "daily_breakdown": daily_costs,
            "budget_status": {
                "daily_budget": self.daily_budget,
                "monthly_budget": self.monthly_budget,
                "daily_remaining": max(0, (self.daily_budget or 0) - self.get_daily_cost()),
                "monthly_remaining": max(0, (self.monthly_budget or 0) - self.get_monthly_cost())
            }
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics in specified format.
        
        Args:
            format: Export format ("json" or "csv")
            
        Returns:
            Formatted metrics data
        """
        if format.lower() == "json":
            return json.dumps({
                "token_usage": [
                    {
                        "timestamp": usage.timestamp,
                        "model": usage.model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "operation_type": usage.operation_type
                    }
                    for usage in self.token_usage_history
                ],
                "performance_metrics": [
                    {
                        "operation_id": metrics.operation_id,
                        "operation_type": metrics.operation_type,
                        "duration_ms": metrics.duration_ms,
                        "success": metrics.success,
                        "timestamp": metrics.start_time
                    }
                    for metrics in self.performance_metrics
                ],
                "daily_stats": dict(self.daily_stats)
            }, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_current_usage(self) -> Dict[str, int]:
        """
        Get current token usage summary.
        
        Returns:
            Dictionary with current usage statistics
        """
        with self._lock:
            if not self.token_usage_history:
                return {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            
            total_tokens = sum(usage.total_tokens for usage in self.token_usage_history)
            input_tokens = sum(usage.input_tokens for usage in self.token_usage_history)
            output_tokens = sum(usage.output_tokens for usage in self.token_usage_history)
            
            return {
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
    
    def get_total_usage(self) -> Dict[str, int]:
        """
        Get total token usage across all operations.
        
        Returns:
            Dictionary with total usage statistics
        """
        return self.get_current_usage()
    
    def get_total_cost(self) -> float:
        """
        Get total cost across all operations.
        
        Returns:
            Total cost in USD
        """
        with self._lock:
            return sum(cost.total_cost for cost in self.cost_history)

    def reset_metrics(self, confirm: bool = False):
        """
        Reset all collected metrics (use with caution).
        
        Args:
            confirm: Must be True to actually reset
        """
        if not confirm:
            raise ValueError("Must confirm reset by setting confirm=True")
        
        with self._lock:
            self.token_usage_history.clear()
            self.cost_history.clear()
            self.performance_metrics.clear()
            self.active_operations.clear()
            self.daily_stats.clear()
            self.latency_window.clear()
            self.throughput_window.clear()
            
            # Reset budget alerts
            for alert in self.budget_alerts:
                alert.triggered = False
                alert.last_triggered = None
        
        logger.info("All metrics have been reset")