"""Patient agent implementation with structured output."""

from typing import Any, Dict

from psibench.prompts.patient_prompt import create_patient_prompt
from psibench.agents.base import BaseAgent
from psibench.models.roleplay_doh import roleplay_doh_rewrite_response


def _extract_nonempty_content(response: Any, context: str) -> str:
    """Extract a non-empty text response from a model output object."""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                part = item.strip()
                if part:
                    parts.append(part)
            elif isinstance(item, dict):
                text_field = str(item.get("text", "")).strip()
                if text_field:
                    parts.append(text_field)
        text = "\n".join(parts).strip()
        if text:
            return text

    raise ValueError(f"Empty patient response ({context})")


class PatientAgent(BaseAgent):
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
        # Initialize base agent
        super().__init__(config, "patient", model_name)
        
        self.patient_profile = patient_profile
        self.psi = config.get("patient").get("simulator")
        self.prompt = create_patient_prompt(psi=self.psi)
        self.chain = self.prompt | self.llm


    def respond(
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
        try:
            if self.psi == "roleplaydoh":
                # Initial response generation using a basic prompt
                inputs = {
                    "conversation_history": self._format_history(conversation_history),
                    "therapist_message": therapist_message,
                }
                
                initial_response = self.chain.invoke(inputs)

                parsed_response = _extract_nonempty_content(initial_response, "roleplaydoh")

                # Prepare prompts for roleplay_doh_pipeline
                # The pipeline expects a list of dicts with "role" and "content"
                prompts_for_pipeline = conversation_history + [
                    {"role": "user", "content": therapist_message}
                ]

                # Refine the response using roleplay_doh_pipeline
                refined_response = roleplay_doh_rewrite_response(
                    self.llm,  # client
                    prompts_for_pipeline,  # initial_prompts
                    parsed_response,  # response_content
                    self.patient_profile,  # profile
                )
                refined_text = str(refined_response).strip()
                if not refined_text:
                    raise ValueError("Empty patient response (roleplaydoh refined)")
                return refined_text
            
            elif self.psi == "patientpsi":
                #The patient profile consists of the system prompt itself
                inputs = {
                    "system_prompt": self.patient_profile,
                    "conversation_history": self._format_history(conversation_history),
                    "therapist_message": therapist_message,
                }
                response = self.chain.invoke(inputs)
                return _extract_nonempty_content(response, "patientpsi")
            else:
                pass
        except Exception:
            raise
            
