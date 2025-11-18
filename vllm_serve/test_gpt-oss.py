import litellm

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = litellm.completion(
            model="hosted_vllm/openai/gpt-oss-120b", # pass the vllm model name
            messages=messages,
            api_base="http://141.142.254.223:8000/v1",
            reasoning_effort="medium", # can be "low", "medium", "high"
            temperature=0.9,
            max_tokens=2000)

content = response.choices[0].message.content
reasoning = response.choices[0].message.reasoning_content
print("\nREASONING:")
print(reasoning)
print("RESPONSE:")
print(content)