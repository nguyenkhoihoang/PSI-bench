"""Base agent implementation with common functionality."""

from typing import Dict, Any
from langchain_litellm import ChatLiteLLM
import logging

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

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
            max_tokens=config.get(agent_type).get("max_tokens", 2000),
        )
    
    def _format_history(self, history: list[Dict[str, str]], max_history_tokens: int = 6000) -> str:
        """Format conversation history for the prompt.
        
        Args:
            history: List of conversation messages
            max_history_tokens: Maximum approximate tokens for history (default: 6000)
            
        Returns:
            Formatted conversation history string, truncated if necessary
        """
        if not history:
            return "Beginning of session."
        
        # Truncate history if it's too long to prevent context overflow
        # Keep recent messages (estimate ~4 chars per token)
        truncated_history = history
        if len(history) > 10:  # Only truncate if we have many messages
            estimated_tokens = sum(len(msg.get("content", "")) for msg in history) // 4
            if estimated_tokens > max_history_tokens:
                # Keep last N messages that fit within token limit
                truncated_history = []
                current_tokens = 0
                for msg in reversed(history):
                    msg_tokens = len(msg.get("content", "")) // 4
                    if current_tokens + msg_tokens > max_history_tokens:
                        break
                    truncated_history.insert(0, msg)
                    current_tokens += msg_tokens
        formatted = []
        if len(truncated_history) < len(history):
            formatted.append("[Earlier messages truncated...]")
        for msg in truncated_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
