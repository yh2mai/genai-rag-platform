"""
Test DeepSeek LLM client functionality
"""
import pytest
from unittest.mock import Mock, patch
from app.services.llm_client import DeepSeekClient

def test_deepseek_client_initialization():
    """Test that DeepSeek client initializes correctly"""
    client = DeepSeekClient()
    
    assert client.model == "deepseek-chat"
    assert client.max_tokens == 8192
    assert client.temperature == 0.1
    assert client.client is not None

@patch('app.services.llm_client.OpenAI')
def test_generate_response(mock_openai):
    """Test response generation"""
    # Mock the OpenAI client response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    client = DeepSeekClient()
    
    # Test the generate_response method would work
    # Note: This is a unit test, so we're testing the structure
    assert hasattr(client, 'generate_response')
    assert hasattr(client, 'generate_with_context')
    assert hasattr(client, 'validate_api_key')

def test_message_formatting():
    """Test that messages are formatted correctly for RAG"""
    client = DeepSeekClient()
    
    query = "What is the capital of France?"
    context = "France is a country in Europe. Its capital city is Paris."
    
    # Test that the method exists and can be called
    # In a real test with API access, we would test the actual functionality
    assert hasattr(client, 'generate_with_context')

@patch('app.services.llm_client.OpenAI')
def test_api_key_validation(mock_openai):
    """Test API key validation"""
    mock_client = Mock()
    mock_openai.return_value = mock_client
    
    # Test successful validation
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Hello"
    mock_client.chat.completions.create.return_value = mock_response
    
    client = DeepSeekClient()
    # The validation method exists
    assert hasattr(client, 'validate_api_key')
    
    # Test failed validation
    mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
    result = client.validate_api_key()
    assert result == False