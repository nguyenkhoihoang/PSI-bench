"""Therapist agent implementation with structured output."""

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_litellm import ChatLiteLLM
from psibench.prompts.therapist_prompt import create_therapist_prompt
from dotenv import load_dotenv
from psibench.utils.llm_backend import (
    parse_model_response,
    prompt_to_openai_messages,
)

load_dotenv()

class TherapistResponse(BaseModel):
    """Structured output schema for therapist responses."""
    
    response: str = Field(description="The therapist's response to the patient")


class TherapistAgent:
    """LLM-based therapist agent to interact with patient simulator."""
    
    def __init__(self,  config: Dict[str, Any], model_name: str = None, model_interface_client = None):
        """
        Initialize the therapist agent.
        
        Args:
            model_name: OpenAI model to use (default: gpt-4.1-mini)
        """
        self.config = config
        therapist_cfg = config.get("therapist", {})

        self.temperature = therapist_cfg.get("temperature", 0.5)
        self.top_p = therapist_cfg.get("top_p", 1.0)
        self.max_tokens = therapist_cfg.get("max_tokens", 512)
        self.model_kwargs = therapist_cfg.get("model_kwargs", {})
        self.backend = therapist_cfg.get("backend", "langchain").lower()
        self.model_interface_config = (
            therapist_cfg.get("model_interface_config")
            or config.get("model_interface", {}).get("config_path")
        )
        self.prompt = create_therapist_prompt()

        # Use model from config if not explicitly provided
        if model_name is None:
            model_name = therapist_cfg.get("model")

        if self.backend == "model_interface":
            if not self.model_interface_config and model_interface_client is None:
                raise ValueError(
                    "TherapistAgent backend is set to model_interface but no config path was provided."
                )
            if model_interface_client is not None:
                self.model_interface_client = model_interface_client
            else:
                from model_interface import create_model
                self.model_interface_client = create_model(self.model_interface_config)
            self.chain = None
        else:
            self.llm = ChatLiteLLM(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                temperature=self.temperature,
            )
            self.chain = self.prompt | self.llm.with_structured_output(TherapistResponse)
        self._generation_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            **(self.model_kwargs or {}),
        }
    
    def respond(self, conversation_history: list[Dict[str, str]], patient_message: str = None) -> str:
        """
        Generate a therapeutic response.
        
        Args:
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            patient_message: The patient's latest message (None for starting the conversation)
            
        Returns:
            The therapist's response
        """
        inputs = self._build_inputs(conversation_history, patient_message)
        if self.backend == "model_interface":
            return self._respond_with_model_interface(inputs)

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

    def build_inputs(self, conversation_history: List[Dict[str, str]], patient_message: str = None) -> Dict[str, Any]:
        return self._build_inputs(conversation_history, patient_message)

    def build_model_interface_messages(self, inputs: Dict[str, Any]) -> List[Dict[str, str]]:
        return prompt_to_openai_messages(self.prompt, inputs)

    def parse_model_interface_response(self, raw: str) -> str:
        parsed = parse_model_response(raw, TherapistResponse)
        return parsed.response

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        return dict(self._generation_kwargs)

    def _respond_with_model_interface(self, inputs: Dict[str, Any]) -> str:
        """Call the unified model interface backend."""
        if not self.model_interface_client:
            raise RuntimeError("Model interface client is not initialized.")
        messages = self.build_model_interface_messages(inputs)
        raw_response = self.model_interface_client.generate(
            [messages],
            **self._generation_kwargs,
        )[0]
        return self.parse_model_interface_response(raw_response)

    def _build_inputs(self, conversation_history: List[Dict[str, str]], patient_message: str = None) -> Dict[str, Any]:
        return {
            "conversation_history": self._format_history(conversation_history),
            "patient_message": patient_message or "Starting the session"
        }
