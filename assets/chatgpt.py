import openai

openai.api_key = "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"

def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"].strip()


prompt = input("Enter your prompt: ")
print("- - - - - - - - - - - - - - - - - - - - - - - - - -")
gpt_response = get_gpt_response(prompt)
print("GPT Response:\n", gpt_response)
