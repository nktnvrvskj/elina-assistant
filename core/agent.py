"""ИИ с инструментами: модель просит JSON-вызов, агент выполняет и
возвращает результат обратно модели.

Инструменты описываются в системном промпте, поэтому работают с любой
моделью в llama-server — без нативного tool-calling.
"""

from __future__ import annotations

import ast
import atexit
import json
import re

from core.tools import Tools

INSTRUCTIONS = """
У тебя есть инструменты:
{tools}

Если для ответа нужен инструмент, ответь РОВНО одной строкой JSON, без
пояснений, без markdown и без текста после неё:
{{"tool": "имя_инструмента", "args": {{"параметр": "значение"}}}}

В args — только параметры из описания инструмента. Если параметров нет,
оставь пустой объект: "args": {{}}.

На приветствия, «как дела», «спасибо» и болтовню инструменты не нужны —
отвечай обычным текстом.

Вызывай один инструмент за раз. Если инструмент не нужен, отвечай обычным
текстом. Никогда не выдумывай результаты инструментов — дождись их.
""".strip()

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


def _norm(name) -> str:
    """«battery()» и «Battery» → «battery»."""
    return re.sub(r"[^0-9a-zа-я_.]", "", str(name).lower())


def _clean(text: str) -> str:
    """Убирает служебные теги размышлений, если модель их всё же выдала."""
    if not text:
        return ""
    text = text.replace("\r", "")
    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        text = head.strip() or tail
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _repair_json(text: str) -> str:
    """Чинит частые ошибки маленьких моделей: лишние кавычки и запятые."""
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r'(?<=[0-9}\]])\s*"(?=\s*[}\]])', "", text)
    return text


def _loads_object(text: str) -> dict | None:
    """Сначала обычный JSON, потом починенный, потом python-словарь."""
    for candidate in (text, _repair_json(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(data, dict):
            return data
    return None


def _extract_args(text: str) -> dict:
    """Достаёт объект args, даже если остальной JSON сломан."""
    match = re.search(r'"(?:args|arguments)"\s*:\s*\{', text)
    if not match:
        return {}
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        symbol = text[index]
        if symbol == "{":
            depth += 1
        elif symbol == "}":
            depth -= 1
            if depth == 0:
                data = _loads_object(text[start:index + 1])
                return data if isinstance(data, dict) else {}
    return {}


def parse_tool_call(reply: str, names: set[str]) -> dict | None:
    """Ищет в ответе модели JSON вида {"tool": ..., "args": {...}}."""
    if not reply:
        return None
    text = reply.replace("\r", "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]

    mapping = {_norm(name): name for name in names}

    def make_call(data: dict) -> dict | None:
        raw = data.get("tool") or data.get("name") or data.get("function")
        real = mapping.get(_norm(raw)) if isinstance(raw, str) else None
        if not real:
            return None
        args = data.get("args") or data.get("arguments")
        if not isinstance(args, dict):
            args = {}
        return {"tool": real, "args": args}

    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            symbol = text[index]
            if symbol == "{":
                depth += 1
            elif symbol == "}":
                depth -= 1
                if depth == 0:
                    data = _loads_object(text[start:index + 1])
                    if data is not None:
                        call = make_call(data)
                        if call:
                            return call
                    break
        start = text.find("{", start + 1)

    # JSON совсем сломан, но имя инструмента видно — вытаскиваем его.
    match = re.search(r"['\"]tool['\"]\s*:\s*['\"]([^'\"]+)", text)
    if match:
        real = mapping.get(_norm(match.group(1)))
        if real:
            return {"tool": real, "args": _extract_args(text)}
    return None


def _looks_like_call(reply: str) -> bool:
    """Похоже, что модель пыталась вызвать инструмент, но не смогла."""
    text = reply.replace("`", "").lstrip()
    return (text.startswith("{") and "tool" in text
            and ("args" in text or "arguments" in text))


class Agent:
    """Диалог с моделью, которая может вызывать инструменты."""

    def __init__(self, config, api, tools: Tools | None = None,
                 max_steps: int = 3) -> None:
        self.config = config
        self.api = api
        self.tools = tools if tools is not None else Tools(config)
        self.max_steps = max_steps
        self._system: str | None = None
        atexit.register(self.close)

    def close(self) -> None:
        self.tools.close()

    def base_prompt(self) -> str:
        return (getattr(self.config, "SYSTEM_PROMPT", "") or "").strip()

    def system_prompt(self) -> str:
        if self._system is None:
            base = self.base_prompt()
            tools_hint = self.tools.hint()
            extra = INSTRUCTIONS.format(tools=tools_hint) if tools_hint else ""
            self._system = f"{base}\n\n{extra}".strip() if extra else base
        return self._system

    def _chat(self, text: str) -> str:
        """Обычный разговор без инструментов: привет, «как дела», спасибо."""
        messages = [
            {"role": "system", "content": self.base_prompt()},
            {"role": "user", "content": text},
        ]
        reply = _clean(self.api.send_messages(messages) or "")
        if not reply or _looks_like_call(reply):
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    "Ответь обычным текстом, без JSON и без вызова инструментов."
                )},
            ]
            reply = _clean(self.api.send_messages(messages) or "")
        if not reply or _looks_like_call(reply):
            return _fallback(text)
        return reply

    def ask(self, text: str) -> str:
        """Ответ на вопрос; при необходимости сначала вызывает инструмент."""
        if _is_smalltalk(text):
            return self._chat(text)

        system = {"role": "system", "content": self.system_prompt()}
        messages = [system, {"role": "user", "content": text}]
        names = self.tools.names()

        reply = _clean(self.api.send_messages(messages) or "")
        result = ""

        for step in range(self.max_steps):
            call = parse_tool_call(reply, names)
            if call is None:
                if _looks_like_call(reply):
                    if step == 0:
                        messages = messages + [
                            {"role": "assistant", "content": reply},
                            {"role": "user", "content": (
                                "JSON не разобрался. Пришли корректный JSON "
                                "одной строкой или ответь обычным текстом."
                            )},
                        ]
                        reply = _clean(self.api.send_messages(messages) or "")
                        continue
                    return ("не смог разобрать вызов инструмента, "
                            "попробуйте переформулировать")
                return reply or result or "не смог ответить, попробуйте переформулировать"

            result = self.tools.call(call["tool"], call["args"])

            # В историю кладём канонический JSON, а не сырой ответ модели:
            # у маленьких моделей там часто мусор.
            messages = messages[:2] + [
                {"role": "assistant", "content": json.dumps(call, ensure_ascii=False)},
                {"role": "user", "content": (
                    f"Результат инструмента: {result}\n\n"
                    "Ответь пользователю коротко, по-русски, без JSON, "
                    "опираясь на этот результат."
                )},
            ]
            reply = _clean(self.api.send_messages(messages) or "")

        if _looks_like_call(reply):
            return ("не смог разобрать ответ модели, "
                    "попробуйте переформулировать")
        return reply or result or "не смог ответить, попробуйте переформулировать"
