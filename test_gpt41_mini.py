import litellm
import os
from dotenv import load_dotenv
load_dotenv()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = litellm.completion(
            model="gpt-4.1-mini",
            messages=messages,
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ["OPENAI_BASE_URL"],
            temperature=0.9,
            max_tokens=2000)

content = response.choices[0].message.content

print("RESPONSE:")
print(content)