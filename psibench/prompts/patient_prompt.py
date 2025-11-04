from langchain_core.prompts import ChatPromptTemplate

def create_patient_prompt(psi: str) -> ChatPromptTemplate:
    """
    router
    """
    if psi == "eeyore":
        return create_eeyore_prompt()
    elif psi == "patient_psi":
        return create_patient_psi_prompt()

def create_eeyore_prompt() -> ChatPromptTemplate:
    system_message = """You will act as a help-seeker struggling with negative emotions in a conversation with someone who is listening to you.

    YOUR PROFILE:
    {eeyore_profile}

    YOUR TASK:
    As the client, your role is to continue the conversation by responding naturally to the supporter, reflecting the characteristics outlined in your profile.
    """

    human_message = """Conversation so far:
    {conversation_history}

    Therapist: {current_message}

    Respond as the patient:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])
    
def create_patient_psi_prompt() -> ChatPromptTemplate:
    #TODO
    pass