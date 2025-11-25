import litellm

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = litellm.completion(
            model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", # pass the vllm model name
            messages=messages,
            api_base="http://141.142.254.222:9000/v1",
            temperature=0.9,
            max_tokens=2000)

content = response.choices[0].message.content

print("RESPONSE:")
print(content)
