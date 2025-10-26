from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # VLLM doesn't require a real key
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ],
    temperature=0.7,
    # max_tokens=100
)

print(response.choices[0].message.content)