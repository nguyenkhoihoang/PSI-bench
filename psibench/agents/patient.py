"""Patient agent implementation with structured output."""

import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_litellm import ChatLiteLLM
from psibench.prompts.patient_prompt import create_patient_prompt
from dotenv import load_dotenv
from psibench.utils.llm_backend import (
    parse_model_response,
    prompt_to_openai_messages,
)

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
        model_name: str = None,
        model_interface_client = None
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

        patient_cfg = config.get("patient", {})
        self.psi = patient_cfg.get("simulator")
        self.prompt = create_patient_prompt(psi=self.psi)

        self.temperature = patient_cfg.get("temperature", 0.7)
        self.top_p = patient_cfg.get("top_p", 1.0)
        self.max_tokens = patient_cfg.get("max_tokens", 512)
        self.model_kwargs = patient_cfg.get("model_kwargs", {})
        self.backend = patient_cfg.get("backend", "langchain").lower()
        self.model_interface_config = (
            patient_cfg.get("model_interface_config")
            or config.get("model_interface", {}).get("config_path")
        )

        # Use model from config if not explicitly provided
        if model_name is None:
            model_name = patient_cfg.get("model")

        if self.backend == "model_interface":
            if not self.model_interface_config and model_interface_client is None:
                raise ValueError(
                    "PatientAgent backend is set to model_interface but no config path was provided."
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
            self.chain = self.prompt | self.llm.with_structured_output(PatientResponse)
        
        self._generation_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            **(self.model_kwargs or {}),
        }
    
    def respond(self, conversation_history: list[Dict[str, str]], therapist_message:str) -> str:
        """
        Generate a response to the therapist's message.
        
        Args:
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            therapist_message: The therapist's latest message
            
        Returns:
            The patient's response
        """
        inputs = self._build_inputs(conversation_history, therapist_message, log_prompt=True)

        if self.backend == "model_interface":
            return self._respond_with_model_interface(inputs)

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

    def build_inputs(self, conversation_history: List[Dict[str, str]], therapist_message: str) -> Dict[str, Any]:
        """Public helper for batching."""
        return self._build_inputs(conversation_history, therapist_message, log_prompt=False)

    def build_model_interface_messages(self, inputs: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return OpenAI-format messages for model_interface batching."""
        return prompt_to_openai_messages(self.prompt, inputs)

    def parse_model_interface_response(self, raw: str) -> str:
        """Parse raw model_interface output into patient text."""
        parsed = parse_model_response(raw, PatientResponse)
        return parsed.response

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        return dict(self._generation_kwargs)

    def _build_inputs(
        self,
        conversation_history: List[Dict[str, str]],
        therapist_message: str,
        log_prompt: bool = False,
    ) -> Dict[str, Any]:
        if self.psi == "eeyore":
            inputs = {
                "eeyore_system_prompt": self.patient_profile.get("eeyore_system_prompt", ""),
                "conversation_history": self._format_history(conversation_history),
                "therapist_message": therapist_message
            }
            if log_prompt:
                print(self.prompt.format(**inputs))
        else:
            inputs = {
                "conversation_history": self._format_history(conversation_history),
                "therapist_message": therapist_message
            }
        return inputs

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
