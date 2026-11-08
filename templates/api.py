import json
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = """
Ты краткий разговорный помощник.
Отвечай сразу по существу.
Не показывай ход рассуждений.
Не пиши длинных вступлений.
Обычно отвечай 1-3 предложениями.
"""


def ask_ai(message):
    data = {
        "model": "LFM3-1.7B",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0.3,
        "top_k": 40,
        "max_tokens": 256,
        "stream": False
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["choices"][0]["message"]["content"]

    except urllib.error.URLError as error:
        return f"Ошибка подключения к ИИ: {error}"

    except (KeyError, IndexError, json.JSONDecodeError) as error:
        return f"Ошибка ответа модели: {error}"


def main():
    print("LFM3-1.7B запущена!")
    print("Напиши exit для выхода.\n")

    while True:
        message = input("Ты: ").strip()

        if message.lower() in ("exit", "выход"):
            break

        if not message:
            continue

        answer = ask_ai(message)
        print(f"\nИИ: {answer}\n")


if __name__ == "__main__":
    main()
