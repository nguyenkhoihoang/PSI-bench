"""
vLLM Offline provider model implementation.
"""
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from tqdm import tqdm
from ..base.model import BaseModel
from .model_config import VLLMOfflineConfig


class VLLMOfflineModel(BaseModel):
    """vLLM Offline model provider implementation using direct LLM."""

    def __init__(self, config: VLLMOfflineConfig):
        """
        Initialize vLLM Offline model.

        Args:
            config: vLLM Offline configuration instance
        """
        super().__init__(config)
        self.config: VLLMOfflineConfig = config

        # Initialize vLLM LLM
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
            tokenizer_mode=config.tokenizer_mode,
            max_model_len=config.max_model_len,
        )

        # Get tokenizer for message conversion
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of message lists using vLLM's batch generation.

        Args:
            messages_list: List of message lists in OpenAI format
            temperature: Temperature parameter (default: 0.7)
            top_p: Top-p sampling parameter (default: 1.0)
            max_tokens: Maximum tokens to generate (default: 2048)
            **kwargs: Additional vLLM sampling parameters

        Returns:
            List of generated response strings
        """
        if not messages_list:
            return []

        # Convert all message lists to prompts
        prompts = []
        for messages in tqdm(messages_list, desc="Converting messages"):
            prompt = self.convert_messages(messages)
            prompts.append(prompt)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )

        # Generate using vLLM's batch generation (vLLM shows its own progress bar)
        print("Generating responses with vLLM...")
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)

        # Extract generated text from outputs
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        return results

    def convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI format messages to vLLM prompt format using chat template.

        Args:
            messages: List of messages in OpenAI format
                     Example: [{"role": "user", "content": "Hello"}]

        Returns:
            Formatted prompt string
        """
        # Check if tokenizer has apply_chat_template method
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Use the model's chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise NotImplementedError(
                "The tokenizer does not support chat templates. "
                "Please implement a custom conversion method."
            )

        return prompt
