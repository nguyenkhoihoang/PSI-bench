"""Base agent implementation with common functionality."""

from typing import Dict, Any
from langchain_litellm import ChatLiteLLM


class BaseAgent:
    """Base class for LLM-based agents with common functionality."""
    
    def __init__(self, config: Dict[str, Any], agent_type: str, model_name: str = None):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration dictionary
            agent_type: Type of agent ('patient' or 'therapist')
            model_name: Optional model override, if not specified uses config
        """
        self.config = config
        self.agent_type = agent_type
        
        # Use model from config if not explicitly provided
        if model_name is None:
            model_name = config.get(agent_type).get("model")
        
        # Get API base and key
        if config.get(agent_type).get("api_base"):
            api_base = config.get(agent_type).get("api_base")
            api_key = "sk-no-key-required"
        else:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_base = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize LLM
        self.llm = ChatLiteLLM(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=config.get(agent_type).get("temperature"),
        )
    
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted conversation history string
        """
        if not history:
            return "Beginning of session."
        
        formatted = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
