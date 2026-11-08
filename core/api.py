import json
import urllib.error
import urllib.request


class Api:
    """Запросы к llama-server через OpenAI-совместимый API."""

    # Рекомендованные параметры LFM3 для чата и tool calling.
    # В llama.cpp штраф повторов называется repeat_penalty.
    temperature = 0.1
    top_k = 50
    repeat_penalty = 1.1
    max_tokens = 512

    def __init__(self, config):
        self.config = config

    def complete(self, messages, tools=None, max_tokens=None, thinking=None):
        """Ответ модели: {"content", "tool_calls", "error"}.

        thinking=False — не рассуждать (быстро, для инструментов);
        thinking=True — рассуждать в пределах thinking_budget.
        """
        if thinking is None:
            thinking = bool(getattr(self.config, "thinking", False))
        limit = int(getattr(self.config, "thinking_budget", 128) or 0)
        thinking = bool(thinking) and limit != 0

        budget = max_tokens or self.max_tokens
        if thinking and limit > 0:
            # Оставляем место для ответа после рассуждений.
            budget = max(budget, limit + 256)

        for attempt in range(2):
            result = self._post(messages, budget, tools, thinking)
            if result.get("error"):
                return {"content": "", "tool_calls": [], "error": result["error"]}

            content = (result.get("content") or "").strip()
            calls = result.get("tool_calls") or []
            if content or calls:
                return {"content": content, "tool_calls": calls, "error": None}

            # Пустой ответ при исчерпанном бюджете — пробуем ещё раз.
            if result.get("finish_reason") == "length" and attempt == 0:
                budget *= 2
                continue
            break

        return {"content": "", "tool_calls": [],
                "error": "модель вернула пустой ответ"}

    def _post(self, messages, max_tokens, tools=None, thinking=False):
        url = (f"http://{self.config.host}:{self.config.port}"
               "/v1/chat/completions")
        data = {
            "model": self.config.model or "local",
            "messages": messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking:
            data["chat_template_kwargs"] = {"enable_thinking": True}
            limit = int(getattr(self.config, "thinking_budget", 128) or 0)
            if limit > 0:
                data["reasoning_budget_tokens"] = limit
                data["reasoning_budget_message"] = (
                    "Хватит рассуждать, ответь кратко.")
        else:
            data["chat_template_kwargs"] = {"enable_thinking": False}
            data["reasoning_effort"] = "none"
        if tools:
            data["tools"] = tools

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
            message = choice.get("message") or {}
            return {
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls") or [],
                "finish_reason": choice.get("finish_reason"),
            }

        except urllib.error.URLError as error:
            return {"error": f"Ошибка подключения к ИИ: {error}"}

        except (KeyError, IndexError, json.JSONDecodeError) as error:
            return {"error": f"Ошибка ответа модели: {error}"}
