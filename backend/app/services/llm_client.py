"""
LLM Client for DeepSeek API integration
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.core.config import settings
from app.services.cost_performance_tracker import CostPerformanceTracker

logger = logging.getLogger(__name__)

class DeepSeekClient:
    """Client for interacting with DeepSeek API"""
    
    def __init__(self, cost_tracker: Optional[CostPerformanceTracker] = None):
        """Initialize the DeepSeek client"""
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        self.cost_tracker = cost_tracker
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a response using DeepSeek API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate (defaults to config value)
            temperature: Sampling temperature (defaults to config value)
            stream: Whether to stream the response
            
        Returns:
            Generated response text
        """
        operation_id = str(uuid.uuid4())
        
        # Start performance tracking
        if self.cost_tracker:
            self.cost_tracker.start_operation(operation_id, "llm_generation")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                
                # Track token usage for streaming (estimate based on response length)
                if self.cost_tracker:
                    input_tokens = self._estimate_tokens(messages)
                    output_tokens = self._estimate_tokens([{"role": "assistant", "content": full_response}])
                    
                    usage = self.cost_tracker.track_token_usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=self.model,
                        operation_type="llm_generation"
                    )
                    self.cost_tracker.calculate_cost(usage)
                    self.cost_tracker.end_operation(operation_id, "llm_generation", success=True)
                
                return full_response
            else:
                response_text = response.choices[0].message.content
                
                # Track token usage and cost
                if self.cost_tracker and hasattr(response, 'usage'):
                    usage = self.cost_tracker.track_token_usage(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        model=self.model,
                        operation_type="llm_generation"
                    )
                    self.cost_tracker.calculate_cost(usage)
                    self.cost_tracker.end_operation(operation_id, "llm_generation", success=True)
                
                return response_text
                
        except Exception as e:
            logger.error(f"Error generating response with DeepSeek: {e}")
            
            # Track failed operation
            if self.cost_tracker:
                self.cost_tracker.end_operation(
                    operation_id, 
                    "llm_generation", 
                    success=False, 
                    error_message=str(e)
                )
            
            raise
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate token count for messages (rough approximation).
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Rough approximation: 1 token ≈ 4 characters for English text
        return max(1, total_chars // 4)
    
    async def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response with context (for RAG)
        
        Args:
            query: User query
            context: Retrieved context from documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": (
                    "You are an intelligent assistant that answers questions based on provided context. "
                    "Use only the information from the context to answer questions. "
                    "If the context doesn't contain enough information, say so clearly. "
                    "Always cite your sources when possible."
                )
            })
        
        # Add context and query
        user_message = f"""Context:
{context}

Question: {query}

Please answer the question based on the provided context."""
        
        messages.append({"role": "user", "content": user_message})
        
        return await self.generate_response(messages)
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is configured and working
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Make a simple test call to validate the API key
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "model": self.model,
            "base_url": settings.LLM_BASE_URL,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        } 