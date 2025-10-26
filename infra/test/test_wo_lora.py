from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # VLLM doesn't require a real key
)


# ****Change model_id
response = client.chat.completions.create(
    model="82d0f1ec-7",
    messages=[
        {"role": "user", "content": "How did Eugene's activities on Spotify and YouTube relate to each other?"}
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    temperature=0.05,
    top_p=0.95,
    max_tokens=1000,
    # 6c1492fc-2
)

print(response.choices[0].message.content)