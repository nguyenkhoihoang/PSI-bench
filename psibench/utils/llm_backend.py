"""Utility helpers for switching between LangChain and model_interface backends."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Type

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError


def prompt_to_openai_messages(
    prompt: ChatPromptTemplate, inputs: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Render a ChatPromptTemplate and convert it into OpenAI-style messages."""
    formatted_messages = prompt.format_messages(**inputs)
    return [_convert_message(msg) for msg in formatted_messages]


def parse_model_response(
    raw_response: str, response_model: Type[BaseModel]
) -> BaseModel:
    """Best-effort parser that converts raw strings into the desired pydantic model."""
    if not issubclass(response_model, BaseModel):
        raise TypeError("response_model must inherit from BaseModel")

    try:
        return response_model.model_validate_json(raw_response)
    except (ValidationError, json.JSONDecodeError, ValueError):
        pass

    try:
        data = json.loads(raw_response)
        if not isinstance(data, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        data = {"response": raw_response}

    if "response" not in data:
        data["response"] = raw_response

    return response_model.model_validate(data)


def _convert_message(message: BaseMessage) -> Dict[str, str]:
    """Convert LangChain message objects to OpenAI response format."""
    role = getattr(message, "role", None)
    if not role:
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        role = role_map.get(message.type, "user")

    content = message.content
    if isinstance(content, list):
        # Flatten list-based content (e.g., when messages include tool calls)
        flattened = []
        for chunk in content:
            if isinstance(chunk, dict):
                flattened.append(chunk.get("text", ""))
            else:
                flattened.append(str(chunk))
        content = " ".join(filter(None, flattened))

    return {"role": role, "content": content}
