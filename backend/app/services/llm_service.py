"""
LLM Service for generating responses using DeepSeek API
"""
import openai
from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM service"""
    content: str
    model: str
    tokens_used: Dict[str, int]
    finish_reason: str


class LLMService:
    """Service for interacting with LLM APIs"""
    
    def __init__(self):
        """Initialize LLM service with DeepSeek configuration"""
        self.client = openai.OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        
        logger.info(f"Initialized LLM service with model: {self.model}")
    
    def generate_response(self, 
                         query: str, 
                         context: str,
                         system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generate a response using the LLM
        
        Args:
            query: User's question
            context: Retrieved document context
            system_prompt: Optional system prompt override
            
        Returns:
            LLMResponse with generated content
        """
        try:
            # Check if we have a valid API key
            if not self.client.api_key or self.client.api_key == "test-key-for-development-only":
                logger.warning("No valid DeepSeek API key configured, using fallback response")
                # Create a meaningful response based on the context
                if "No relevant documents found" in context:
                    content = "I don't have any relevant documents in my knowledge base to answer your question. Please upload some documents first, or try rephrasing your question."
                else:
                    # Extract key information from context and create a basic response
                    content = f"Based on the documents I found, here's what I can tell you:\n\n{context}\n\nNote: This is a basic response. For more sophisticated analysis, please configure a valid DeepSeek API key."
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    tokens_used={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    finish_reason="fallback"
                )
            
            # Default system prompt for RAG
            if system_prompt is None:
                system_prompt = """You are a helpful AI assistant that answers questions based on provided document context. 

Instructions:
- Answer the user's question using ONLY the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive in your response
- Cite specific information from the context when relevant
- Do not make up information that isn't in the context
- If asked about topics not covered in the context, politely explain that you can only answer based on the provided documents"""

            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Context from documents:
{context}

Question: {query}

Please answer the question based on the provided context."""}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            finish_reason = response.choices[0].finish_reason
            
            logger.info(f"Generated LLM response: {tokens_used['total_tokens']} tokens used")
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            # Return fallback response with context if available
            if "No relevant documents found" in context:
                content = "I don't have any relevant documents in my knowledge base to answer your question. Please upload some documents first, or try rephrasing your question."
            else:
                content = f"I found some relevant information in the documents, but I'm unable to process it fully due to a technical issue. Here's the raw context I found:\n\n{context[:500]}{'...' if len(context) > 500 else ''}\n\nPlease try again later or contact support."
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason="error"
            )
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the provided text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        try:
            messages = [
                {"role": "system", "content": f"Summarize the following text in no more than {max_length} characters. Be concise but capture the key points."},
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=min(self.max_tokens, max_length // 2),  # Rough estimate
                temperature=0.3,  # Lower temperature for summaries
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract key terms from text
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        try:
            messages = [
                {"role": "system", "content": f"Extract the {max_keywords} most important keywords or key phrases from the following text. Return them as a comma-separated list."},
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.1,  # Very low temperature for keyword extraction
                stream=False
            )
            
            keywords_text = response.choices[0].message.content
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []


# Global LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service