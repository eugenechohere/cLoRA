from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # VLLM doesn't require a real key
)


#         151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# ****Change model_id
response = client.chat.completions.create(
    model="11213293-b",
    messages=[
        {"role": "user", "content": "hey hey, do you know which  text Eugene highlighte in the expanded description, and what might that suggest?"}
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    temperature=0.05,
    top_p=0.95,
    max_tokens=200,
    # 6c1492fc-2
)

print(response.choices[0].message.content)