"""ИИ с инструментами: LFM3 вызывает их нативно через llama-server.

Инструменты уходят в запрос как tools (OpenAI-совместимо), модель отвечает
tool_calls, агент выполняет вызов и присылает результат ролью tool.
"""

from __future__ import annotations

import atexit
import json
import re

from core.tools import Tools

# Напоминание модели, когда сохранять факты о пользователе.
MEMORY_RULE = (
    "Если пользователь сообщает факт о себе (имя, город, животные, работа, "
    "привычки) — сначала вызови remember и только потом отвечай. Факт "
    "формулируй кратко, по-русски. Не говори, что запомнил, пока инструмент "
    "не вернул результат."
)

# Числа не выдумываем: только из результата инструмента.
DATA_RULE = (
    "Если вопрос про погоду, время, заряд батареи, место на диске или uptime — "
    "сначала вызови соответствующий инструмент. Числа бери только из его "
    "результата, не по памяти."
)

# Короткие фразы для болтовни: инструменты тут не подключаем вообще.
SMALLTALK = {
    "привет", "приветик", "здравствуй", "здравствуйте", "хай", "hello", "hi",
    "доброе утро", "добрый день", "добрый вечер",
    "как дела", "как у тебя дела", "как ты", "как настроение", "как жизнь",
    "что делаешь", "чем занимаешься",
    "спасибо", "спасибо большое", "благодарю", "пока", "до свидания",
}


def _is_smalltalk(text: str) -> bool:
    """True, если всё сообщение — приветствие/благодарность/прощание."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я\s]", " ", text)
    words = [word for word in text.split() if word not in ("элина", "елина")]
    return " ".join(words) in SMALLTALK


def _fallback(text: str) -> str:
    """Ответ на болтовню, если модель так и не смогла ответить текстом."""
    text = text.lower().replace("ё", "е")
    if "спасибо" in text or "благодар" in text:
        return "Пожалуйста!"
    if "пока" in text or "свидания" in text:
        return "До встречи!"
    return "Всё хорошо! Чем помочь?"


def _clean(text: str) -> str:
    """Убирает служебные теги размышлений, если модель их всё же выдала."""
    if not text:
        return ""
    text = text.replace("\r", "")
    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        text = head.strip() or tail
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _looks_like_call(reply: str) -> bool:
    """Похоже, что модель пыталась вызвать инструмент, но не смогла."""
    text = reply.replace("`", "").lstrip()
    return (text.startswith("{") and "tool" in text
            and ("args" in text or "arguments" in text))


def _direct_tool(text: str):
    """Простые команды без аргументов: вызываем инструмент сразу."""
    low = text.lower().replace("ё", "е")
    if any(key in low for key in (
            "который час", "сколько времени", "текущее время",
            "какое сейчас время", "какое число", "какая дата",
            "какой сегодня день", "какой день недели")):
        return "get_time", {}
    if "батаре" in low or "аккумулятор" in low or "заряд" in low:
        return "battery", {}
    if "uptime" in low or "аптайм" in low or "сколько работает" in low:
        return "uptime", {}
    if (("свободн" in low and "мест" in low) or "место на диске" in low
            or "сколько места" in low):
        return "disk_free", {}
    return None


def _arguments(raw) -> dict:
    """Аргументы tool_calls: строка JSON или готовый словарь."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class Agent:
    """Диалог с моделью, которая умеет вызывать инструменты."""

    def __init__(self, config, api, tools: Tools | None = None,
                 max_steps: int = 3) -> None:
        self.config = config
        self.api = api
        self.tools = tools if tools is not None else Tools(config)
        self.max_steps = max_steps
        self.memory_turns = max(0, int(getattr(config, "memory_turns", 5)))
        budget = int(getattr(config, "thinking_budget", 128) or 0)
        self.thinking = bool(getattr(config, "thinking", False)) and budget != 0
        self.history: list[dict] = []
        atexit.register(self.close)

    def close(self) -> None:
        self.tools.close()

    def base_prompt(self) -> str:
        return (getattr(self.config, "SYSTEM_PROMPT", "") or "").strip()

    def system_prompt(self, with_tools: bool = True) -> str:
        parts = [self.base_prompt()]
        facts = self.tools.memory.prompt(with_tool=with_tools)
        if facts:
            parts.append(facts)
        if with_tools:
            parts.append(MEMORY_RULE)
            parts.append(DATA_RULE)
        return "\n\n".join(part for part in parts if part)

    def _start_messages(self, text: str) -> list[dict]:
        return ([{"role": "system", "content": self.system_prompt()}]
                + self.history
                + [{"role": "user", "content": text}])

    def _remember(self, text: str, answer: str) -> None:
        if not answer:
            return
        self.history.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ])
        limit = self.memory_turns * 2
        self.history = self.history[-limit:] if limit else []

    def _chat(self, text: str) -> str:
        """Обычный разговор без инструментов: привет, «как дела», спасибо."""
        messages = [
            {"role": "system", "content": self.system_prompt(with_tools=False)},
            {"role": "user", "content": text},
        ]
        messages[1:1] = self.history
        result = self.api.complete(messages, thinking=False)
        reply = _clean(result.get("content") or "")
        if not reply or _looks_like_call(reply):
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    "Ответь обычным текстом, без JSON и без вызова инструментов."
                )},
            ]
            result = self.api.complete(messages)
            reply = _clean(result.get("content") or "")
        if not reply or _looks_like_call(reply):
            reply = _fallback(text)
        self._remember(text, reply)
        return reply

    def ask(self, text: str) -> str:
        """Ответ на вопрос; при необходимости сначала вызывает инструменты."""
        if _is_smalltalk(text):
            return self._chat(text)

        messages = self._start_messages(text)
        tools = self.tools.schemas() or None
        names = {item["function"]["name"] for item in tools or []}

        used_tools = False
        direct = _direct_tool(text)
        if direct and direct[0] in names:
            used_tools = True
            name, args = direct
            output = self.tools.call(name, args)
            call_id = "direct_1"
            messages += [
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name,
                                 "arguments": json.dumps(args, ensure_ascii=False)},
                }]},
                {"role": "tool", "tool_call_id": call_id, "content": str(output)},
            ]

        result = self.api.complete(messages, tools=tools, thinking=False)
        if result.get("error") and not result.get("tool_calls"):
            return result["error"]
        for step in range(self.max_steps):
            calls = result.get("tool_calls") or []
            if not calls:
                answer = _clean(result.get("content") or "")
                if not answer:
                    return result.get("error") or "не смог ответить, попробуйте переформулировать"

                # Рассуждаем только для обычных вопросов: после вызова
                # инструментов ответ уже опирается на реальные данные.
                if self.thinking and not used_tools:
                    better = self.api.complete(messages, thinking=True)
                    better_answer = _clean(better.get("content") or "")
                    if better_answer:
                        answer = better_answer

                self._remember(text, answer)
                return answer

            used_tools = True
            messages.append({
                "role": "assistant",
                "content": result.get("content") or "",
                "tool_calls": calls,
            })
            for index, call in enumerate(calls):
                function = call.get("function") or {}
                name = function.get("name") or ""
                output = self.tools.call(name, _arguments(function.get("arguments")))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or f"call_{index}",
                    "content": str(output),
                })

            result = self.api.complete(messages, tools=tools, thinking=False)

        # Модель зациклилась на вызовах — просим ответить уже текстом.
        result = self.api.complete(messages, tools=None, thinking=False)
        answer = _clean(result.get("content") or "")
        answer = (answer or result.get("error")
                  or "не смог ответить, попробуйте переформулировать")
        self._remember(text, answer)
        return answer
