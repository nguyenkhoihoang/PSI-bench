"""Judge prompts for different evaluation purposes."""

from langchain_core.prompts import ChatPromptTemplate

def create_ptc_judge_conversation_prompt() -> ChatPromptTemplate:
    """Create prompt for PTC classification of entire conversation at once.
    
    Returns:
        ChatPromptTemplate for classifying all patient turns in a conversation
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant performing a linguistic analysis task for a research project on therapeutic conversations. 
This is for academic research, and the content may discuss mental health challenges.
Your task is to classify EACH patient turn from a therapy session into one of four categories:

**P (Problem)**: The patient is expressing:
- Confusion, distress, or emotional pain
- Feeling stuck or helpless
- Describing problems without insight
- Negative emotions without perspective
- Complaints or struggles

**T (Transition)**: The patient is showing:
- Beginning to reflect on their situation
- Gaining some perspective
- Starting to consider alternatives
- Expressing curiosity or questioning
- Moving from pure distress to thoughtful consideration

**C (Change)**: The patient is demonstrating:
- Emotional resolution or acceptance
- Reframing their situation positively
- New insights or understanding
- Active problem-solving or planning
- Hope, empowerment, or growth mindset
- Clear perspective shift from the problem

**F (Filler)**: The response is filler:
- Contains no meaningful therapeutic content and does not fit into P, T, or C categories
- Is small talk or neutral procedural social responses

Analyze each patient turn carefully, considering both the content and emotional tone.

You must respond with ONLY a JSON array. Each element should have:
- "content": the patient's message
- "classification": one of P, T, C, or F

Output example:
```json
    [
    {{
        "content": "I feel so lost and don't know what to do.",
        "classification": "P"
    }},
    {{
        "content": "But maybe if I try to think differently, I can find a way out.",
        "classification": "T"
    }}
    ]
```json
Do not include any explanation or additional text outside the JSON array."""),
        ("user", """Here is the complete conversation:

{conversation}

Classify each patient turn and return as JSON array:""")
    ])