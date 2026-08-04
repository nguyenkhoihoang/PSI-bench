"""Therapist agent implementation with structured output."""

import time
import traceback
from typing import Dict, Any

from psibench.prompts.therapist_prompt import create_therapist_prompt
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
        
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.chain.invoke(inputs)
                # DEBUG: Print full response object
                # print(f"[DEBUG] Response object type: {type(response)}")
                # print(f"[DEBUG] Response object: {response}")
                # print(f"[DEBUG] Response.__dict__: {response.__dict__ if hasattr(response, '__dict__') else 'N/A'}")
                # print(f"[DEBUG] Has content attr: {hasattr(response, 'content')}")
                # if hasattr(response, 'content'):
                #     print(f"[DEBUG] Content type: {type(response.content)}, repr: {repr(response.content)}")
                #     print(f"[DEBUG] Content length: {len(response.content) if response.content else 0}")
                
                content = response.content.strip()
                # print(f"[DEBUG] Stripped content length: {len(content)}")
                if not content:
                    raise ValueError("Empty therapist response")
                return content
            except Exception as e:
                # Retry on any error with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    error_type = type(e).__name__
                    print(f"{error_type} in therapist.respond: {e}")
                    print(f"Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries reached for {type(e).__name__}: {e}")
                    print(traceback.format_exc())
                    raise
    
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
