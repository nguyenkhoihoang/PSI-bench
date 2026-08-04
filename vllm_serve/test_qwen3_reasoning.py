import litellm
import re


def parse_thinking_response(content: str) -> dict:
    """
    Parse the thinking process and actual response from model output.

    The model outputs thinking process followed by </think> tag, then the actual response.

    Args:
        content: The raw content from response.choices[0].message.content

    Returns:
        A dictionary with 'thinking' and 'response' keys
    """
    # Look for </think> tag that separates thinking from response
    think_end_pattern = r'</think>'
    match = re.search(think_end_pattern, content, re.IGNORECASE)

    if match:
        # Everything before </think> is the thinking process
        thinking = content[:match.start()].strip()
        # Everything after </think> is the actual response
        response_text = content[match.end():].strip()
    else:
        # No </think> tag found, treat entire content as response
        thinking = None
        response_text = content.strip()

    return {
        'thinking': thinking,
        'response': response_text
    }


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = litellm.completion(
            model="hosted_vllm/Qwen/Qwen3-30B-A3B-Thinking-2507", # pass the vllm model name
            messages=messages,
            api_base="http://141.142.254.223:8000/v1",
            reasoning_effort="medium", # can be "low", "medium", "high"
            temperature=0.9,
            max_tokens=2000)

content = response.choices[0].message.content

# Parse thinking and response
parsed = parse_thinking_response(content)

print("=" * 80)
print("REASONING:")
print("=" * 80)
print(parsed['thinking'] if parsed['thinking'] else "No thinking process found")
print("\n" + "=" * 80)
print("RESPONSE:")
print("=" * 80)
print(parsed['response'])
print("\n" + "=" * 80)


