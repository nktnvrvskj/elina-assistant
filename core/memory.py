"""Постоянная память о пользователе: факты в memory.json.

Короткая память диалога живёт в агенте, а сюда попадают только устойчивые
факты: имя, город, животные, привычки. Файл локальный и не коммитится.

    python -m core.memory --list
    python -m core.memory --add "живёт в Казани"
    python -m core.memory --forget "Казани"
    python -m core.memory --clear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_FACTS = 50


class Memory:
    """Список фактов о пользователе с сохранением в JSON."""

    def __init__(self, path):
        self.path = Path(path)
        self._facts: list[str] = []
        self.load()

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._facts = []
            return
        facts = data.get("facts") if isinstance(data, dict) else data
        if not isinstance(facts, list):
            self._facts = []
            return
        self._facts = [str(fact).strip() for fact in facts if str(fact).strip()]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"facts": self._facts}
        text = json.dumps(data, ensure_ascii=False, indent=2)
        self.path.write_text(text + "\n", encoding="utf-8")

    def facts(self) -> list[str]:
        return list(self._facts)

    def add(self, fact) -> str:
        fact = " ".join(str(fact or "").split()).strip(" .")
        if not fact:
            return "пустой факт — не запомнил"
        low = fact.lower()
        if "[" in fact or "не указано" in low or "неизвестно" in low:
            return "пропустил: в факте нет данных"
        for existing in self._facts:
            if existing.lower() == fact.lower():
                return "уже знаю"
        self._facts.append(fact)
        if len(self._facts) > MAX_FACTS:
            self._facts = self._facts[-MAX_FACTS:]
        self.save()
        return f"запомнил: {fact}"

    def forget(self, text) -> str:
        text = str(text or "").strip().lower()
        if not text:
            return "не указано, что забыть"
        before = len(self._facts)
        self._facts = [fact for fact in self._facts if text not in fact.lower()]
        removed = before - len(self._facts)
        if removed:
            self.save()
        return f"забыл фактов: {removed}" if removed else "такого факта нет"

    def clear(self):
        self._facts = []
        self.save()

    def prompt(self, with_tool: bool = True) -> str:
        """Текст для системного промпта (пустая строка, если фактов нет)."""
        if not self._facts:
            return ""
        hint = "используй, не переспрашивай"
        if with_tool:
            hint += "; новые факты сохраняй через remember"
        lines = "\n".join(f"- {fact}" for fact in self._facts)
        return f"Что ты знаешь о пользователе ({hint}):\n{lines}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m core.memory",
        description="Постоянная память о пользователе.")
    parser.add_argument("--file", help="путь к memory.json")
    parser.add_argument("--list", action="store_true", help="показать факты")
    parser.add_argument("--add", metavar="FACT", help="добавить факт")
    parser.add_argument("--forget", metavar="FACT", help="забыть факт по части текста")
    parser.add_argument("--clear", action="store_true", help="удалить все факты")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    memory = Memory(args.file or root / "memory.json")

    if args.add:
        print(memory.add(args.add))
    elif args.forget:
        print(memory.forget(args.forget))
    elif args.clear:
        memory.clear()
        print("память очищена")
    else:
        facts = memory.facts()
        if not facts:
            print("память пуста")
        for index, fact in enumerate(facts, 1):
            print(f"{index}. {fact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
