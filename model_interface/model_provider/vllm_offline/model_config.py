"""
vLLM Offline provider configuration.
"""
from dataclasses import dataclass
from ..base.model_config import ModelConfig


@dataclass
class VLLMOfflineConfig(ModelConfig):
    """Configuration for vLLM offline mode (direct LLM usage)."""

    # Model loading parameters
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    dtype: str = "auto"  # "auto", "float16", "bfloat16"

    # Tokenizer settings
    tokenizer_mode: str = "auto"  # "auto", "slow"

    # Performance settings
    max_model_len: int = None  # Maximum sequence length (None for model default)

    def __post_init__(self):
        """Validate vLLM-specific configuration."""
        if self.dtype not in ["auto", "float16", "bfloat16", "float32"]:
            raise ValueError(
                f"dtype must be one of: auto, float16, bfloat16, float32. Got {self.dtype}"
            )

        if self.gpu_memory_utilization < 0 or self.gpu_memory_utilization > 1:
            raise ValueError(
                f"gpu_memory_utilization must be between 0 and 1, got {self.gpu_memory_utilization}"
            )
