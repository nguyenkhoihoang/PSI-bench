"""Patient agent implementation with structured output."""

import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_litellm import ChatLiteLLM
from prompts.patient_prompt import create_patient_prompt
from dotenv import load_dotenv
load_dotenv()

class PatientResponse(BaseModel):
    """Structured output schema for patient responses."""
    
    response: str = Field(description="The patient's response to the therapist/supporter")


class PatientAgent:
    """LLM-based patient agent that responds based on their profile and conversation history."""
    
    def __init__(
        self,
        patient_profile: Dict[str, Any],
        config: Dict[str, Any],
        model_name: str = None
    ):
        """
        Initialize the patient agent.
        
        Args:
            patient_profile: Patient profile or cognitive model
            config: Configuration dictionary for the simulation
            model_name: Optional model override, if not specified uses config
        """
        self.patient_profile = patient_profile
        self.config = config
        
        # Use model from config if not explicitly provided
        if model_name is None:
            model_name = config.get("patient").get("model")
        
        self.llm = ChatLiteLLM(
            model=model_name,
            api_key= os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=config.get("patient").get("temperature")
        )
        
        self.psi = config.get("patient").get("simulator")        
        self.prompt = create_patient_prompt(psi=self.psi)
        self.chain = self.prompt | self.llm.with_structured_output(PatientResponse)
    
    def respond(self, conversation_history: list[Dict[str, str]], therapist_message: str) -> str:
        """
        Generate a response to the therapist's message.
        
        Args:
            conversation_history: List of previous messages [{"role": "patient/therapist", "content": "..."}]
            therapist_message: The therapist's latest message
            
        Returns:
            The patient's response
        """
        # Prepare inputs based on prompt variant
        if self.psi == "eeyore":
            inputs = {
                "eeyore_profile": self.patient_profile.get("eeyore_profile", ""),
                "conversation_history": self._format_history(conversation_history),
                "current_message": therapist_message
            }
        else:
           pass
        
        response: PatientResponse = self.chain.invoke(inputs)
        return response.response
    
    
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        formatted = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
