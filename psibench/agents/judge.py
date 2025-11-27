"""Base judge agent implementation."""

from typing import Dict, Any
from psibench.agents.base import BaseAgent


class JudgeAgent(BaseAgent):
    """Base class for LLM-based judge agents used in evaluation."""
    
    def __init__(self, judge_config_key: str,config: Dict[str, Any]):
        """Initialize a generic judge.
        
        Args:
            config: Configuration dictionary
            judge_config_key: Key in config['eval'] for this judge's settings
        """
        # Get judge config from eval section
        try:
            judge_config = config.get("eval").get(judge_config_key, {})
        except AttributeError:
            raise ValueError(f"Configuration for judge '{judge_config_key}' not found in eval section.")
        
        # Create a temporary config structure for BaseAgent
        judge_full_config = config.copy()
        judge_full_config["judge"] = {
            "model": judge_config.get("model"),
            "temperature": judge_config.get("temperature"),
        }
        if "api_base" in judge_config:
            judge_full_config["judge"]["api_base"] = judge_config.get("api_base")
        
        # Initialize base agent
        super().__init__(judge_full_config, "judge")
        
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.
        Switch to Therapist/Assistant labels, makes life easier for judges
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted conversation history string
        """
        formatted = []
        for msg in history:
            role = "Therapist" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)