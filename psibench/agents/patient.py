"""Patient agent implementation with structured output."""

import os
from typing import Any, Dict

import openai
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from prompts.patient_prompt import create_patient_prompt
from pydantic import BaseModel, Field

from psibench.models.roleplay_doh import roleplay_doh_rewrite_response

load_dotenv()


class PatientResponse(BaseModel):
    """Structured output schema for patient responses."""

    response: str = Field(
        description="The patient's response to the therapist/supporter"
    )


class PatientAgent:
    """LLM-based patient agent that responds based on their profile and conversation history."""

    def __init__(
        self,
        patient_profile: Dict[str, Any],
        config: Dict[str, Any],
        model_name: str = None,
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
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=config.get("patient").get("temperature"),
        )

        self.psi = config.get("patient").get("simulator")
        self.prompt = create_patient_prompt(psi=self.psi)
        self.chain = self.prompt | self.llm.with_structured_output(PatientResponse)

    async def respond(
        self, conversation_history: list[Dict[str, str]], therapist_message: str
    ) -> str:
        """
        Generate a response to the therapist's message.

        Args:
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            therapist_message: The therapist's latest message

        Returns:
            The patient's response
        """
        # Prepare inputs based on prompt variant
        if self.psi == "eeyore":
            inputs = {
                "eeyore_system_prompt": self.patient_profile.get(
                    "eeyore_system_prompt", ""
                ),
                "conversation_history": self._format_history(conversation_history),
                "therapist_message": therapist_message,
            }
            # print(self.prompt.format(**inputs)) # Commented out print statement
            response: PatientResponse = await self.chain.ainvoke(inputs)
            return response.response
        elif self.psi == "roleplaydoh":

            # Initial response generation using a basic prompt
            inputs = {
                "conversation_history": self._format_history(conversation_history),
                "therapist_message": therapist_message,
            }
            initial_response: PatientResponse = await self.chain.ainvoke(inputs)

            # Prepare prompts for roleplay_doh_pipeline
            # The pipeline expects a list of dicts with "role" and "content"
            prompts_for_pipeline = conversation_history + [
                {"role": "user", "content": therapist_message}
            ]

            # Refine the response using roleplay_doh_pipeline
            refined_response = await roleplay_doh_rewrite_response(
                self.llm,  # client
                prompts_for_pipeline,  # initial_prompts
                initial_response.response,  # response_content
                self.patient_profile,  # profile
            )

            return refined_response
        else:
            pass

    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        formatted = []
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)
