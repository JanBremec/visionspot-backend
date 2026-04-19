import requests

BASE_URL = "https://server"

while True:
    prompt = input(">>> ")
    if prompt.strip() == "exit":
        break

    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "my-model",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        headers={"Content-Type": "application/json"},
        timeout=300
    )

    try:
        print(r.json()["choices"][0]["message"]["content"])
    except Exception:
        print(r.text)