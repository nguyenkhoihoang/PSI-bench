from langchain_core.prompts import ChatPromptTemplate


def create_patient_prompt(psi: str) -> ChatPromptTemplate:
    """
    router
    """
    if psi == "eeyore":
        return create_eeyore_prompt()
    elif psi == "patient_psi":
        return create_patient_psi_prompt()
    elif psi == "roleplaydoh":
        return create_roleplay_doh_prompt()


def create_eeyore_prompt() -> ChatPromptTemplate:
    system_message = """
    {eeyore_system_prompt}
    Conversation so far:
    {conversation_history}

    User's latest message: {therapist_message}

    Respond as the patient:"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
        ]
    )


def create_patient_psi_prompt() -> ChatPromptTemplate:
    # TODO
    pass


def create_roleplay_doh_prompt() -> ChatPromptTemplate:
    system_message = """
    You are a helpful assistant. Generate a patient response based on the conversation history and therapist's message.
    Conversation so far:
    {conversation_history}

    Therapist's latest message: {therapist_message}

    Respond as the patient:"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
        ]
    )
