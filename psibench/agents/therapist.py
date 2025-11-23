"""Therapist agent implementation with structured output."""

from typing import Dict, Any
from prompts.therapist_prompt import create_therapist_prompt
from psibench.agents.base import BaseAgent


class TherapistAgent(BaseAgent):
    """LLM-based therapist agent to interact with patient simulator."""
    
    def __init__(self, config: Dict[str, Any], model_name: str = None):
        """
        Initialize the therapist agent.
        
        Args:
            config: Configuration dictionary
            model_name: Optional model override
        """
        # Initialize base agent
        super().__init__(config, "therapist", model_name)
        
        self.prompt = create_therapist_prompt()
        self.chain = self.prompt | self.llm
    
    def respond(self, conversation_history: list[Dict[str, str]], patient_message: str = None) -> str:
        """
        Generate a therapeutic response.
        
        Args:
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            patient_message: The patient's latest message (None for starting the conversation)
            
        Returns:
            The therapist's response
        """
        inputs = {
            "conversation_history": self._format_history(conversation_history),
            "patient_message": patient_message or "Starting the session"
        }
        response = self.chain.invoke(inputs)
        return response.content.strip()
    
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.
        Switch roles to match prompt expectations (patient as user, therapist as assistant).
        
        Overrides base class method to swap roles for therapist perspective.
        """
        if not history:
            return "Beginning of session."
        
        formatted = []
        for msg in history:
            role = "user" if msg["role"] == "assistant" else "assistant"
            role = role.capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
