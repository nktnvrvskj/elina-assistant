import json
import urllib.error
import urllib.request


class Api:
    """Запросы к llama-server через OpenAI-совместимый API."""

    # LFM3 тратит бюджет на размышления, поэтому thinking выключен.
    max_tokens = 384
    enable_thinking = False

    def __init__(self, config):
        self.config = config

    def send_messages(self, messages, max_tokens=None):
        budget = max_tokens or self.max_tokens
        for attempt in range(2):
            result = self._post(messages, budget)
            if isinstance(result, str):
                return result

            content = (result.get("content") or "").strip()
            if content:
                return content

            # Пустой ответ при исчерпанном бюджете — пробуем ещё раз.
            if result.get("finish_reason") == "length" and attempt == 0:
                budget *= 2
                continue
            break

        return "не смог ответить: модель вернула пустой ответ"

    def _post(self, messages, max_tokens):
        url = (f"http://{self.config.host}:{self.config.port}"
               "/v1/chat/completions")
        data = {
            "model": self.config.model or "local",
            "messages": messages,
            "temperature": 0.3,
            "top_k": 40,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if not self.enable_thinking:
            data["chat_template_kwargs"] = {"enable_thinking": False}

        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))

            choice = result["choices"][0]
            return {
                "content": choice["message"].get("content", ""),
                "finish_reason": choice.get("finish_reason"),
            }

        except urllib.error.URLError as error:
            return f"Ошибка подключения к ИИ: {error}"

        except (KeyError, IndexError, json.JSONDecodeError) as error:
            return f"Ошибка ответа модели: {error}"
