import json
import urllib.request
import urllib.error

class Api:
    def __init__(self, config):
        self.config = config
    def send_ai(self,message):
        API_URL = f"http://{self.config.server_host_llama}:{self.config.server_port_llama}/v1/chat/completions"
        data = {
            "model": "LFM3-1.7B",
            "messages": [
                {
                    "role": "system",
                    "content": self.config.SYSTEM_PROMPT
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
