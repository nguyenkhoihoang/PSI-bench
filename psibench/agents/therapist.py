"""Therapist agent implementation with structured output."""

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_litellm import ChatLiteLLM
from prompts.therapist_prompt import create_therapist_prompt
from dotenv import load_dotenv
load_dotenv()

class TherapistResponse(BaseModel):
    """Structured output schema for therapist responses."""
    
    response: str = Field(description="The therapist's response to the patient")


class TherapistAgent:
    """LLM-based therapist agent to interact with patient simulator."""
    
    def __init__(self,  config: Dict[str, Any], model_name: str = None):
        """
        Initialize the therapist agent.
        
        Args:
            model_name: OpenAI model to use (default: gpt-4.1-mini)
        """
        self.config = config
        
        # Use model from config if not explicitly provided
        if model_name is None:
            model_name = config.get("therapist").get("model")
        
        if config.get("therapist").get("api_base"):
            api_base = config.get("therapist").get("api_base")
            api_key = "sk-no-key-required"
        else:
            api_base = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")
        
        self.llm = ChatLiteLLM(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=config.get("therapist").get("temperature")
        )
         
        self.prompt = create_therapist_prompt()
        self.chain = self.prompt | self.llm.with_structured_output(TherapistResponse)
    
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
        response: TherapistResponse = self.chain.invoke(inputs)
        return response.response
            
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.
        Switch roles to match prompt expectations (patient as user, therapist as assistant).
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
