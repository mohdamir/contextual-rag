import requests

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

payload = {
    "model": "anthropic/claude-3-haiku",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
}

response = requests.post(url, json=payload, headers=headers)
print(response.status_code)
print(response.text)
