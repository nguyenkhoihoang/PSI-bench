"""
OpenAI provider configuration.
"""
from dataclasses import dataclass
from typing import Optional
from ..base.model_config import ModelConfig


@dataclass
class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI API."""

    # For multiprocessing API calls
    num_workers: int = 64