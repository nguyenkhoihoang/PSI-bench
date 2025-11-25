"""Patient agent implementation with structured output."""

from typing import Any, Dict

from psibench.prompts.patient_prompt import create_patient_prompt
from psibench.agents.base import BaseAgent
from psibench.models.roleplay_doh import roleplay_doh_rewrite_response


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
        if self.psi == "eeyore":
            inputs = {
                "eeyore_system_prompt": self.patient_profile.get(
                    "eeyore_system_prompt", ""
                ),
                "conversation_history": self._format_history(conversation_history),
                "therapist_message": therapist_message,
            }
            # print(self.prompt.format(**inputs)) # Commented out print statement
            response = await self.chain.ainvoke(inputs)
            return response.content.strip()
        
        elif self.psi == "roleplaydoh":

            # Initial response generation using a basic prompt
            inputs = {
                "conversation_history": self._format_history(conversation_history),
                "therapist_message": therapist_message,
            }
            initial_response = await self.chain.ainvoke(inputs)
            parsed_response = initial_response.content.strip()

            # Prepare prompts for roleplay_doh_pipeline
            # The pipeline expects a list of dicts with "role" and "content"
            prompts_for_pipeline = conversation_history + [
                {"role": "user", "content": therapist_message}
            ]

            # Refine the response using roleplay_doh_pipeline
            refined_response = await roleplay_doh_rewrite_response(
                self.llm,  # client
                prompts_for_pipeline,  # initial_prompts
                parsed_response,  # response_content
                self.patient_profile,  # profile
            )

            return refined_response
        
        elif self.psi == "patientpsi":
            #The patient profile consists of the system prompt itself
            inputs = {
                "system_prompt": self.patient_profile,
                "conversation_history": self._format_history(conversation_history),
                "therapist_message": therapist_message,
            }
            response = await self.chain.ainvoke(inputs)
            return response.content.strip()
            
        else:
            pass
