"""Первичная настройка: модель, голос, авто-запуск и tools/tools.json.

Запуск: пункт 9 в меню или python main.py --setup.
"""

from __future__ import annotations

import shutil

from core.voice import Voice
from core.wakeword import MODEL_NAME

ENGINES = {"0": "auto", "1": "piper", "2": "espeak-ng", "3": "none"}


def prepare_files(config):
    """Создаёт tools/tools.json и переносит туда старые настройки."""
    old = config.root / "tools.json"
    if old.exists() and not config.tools_file.exists():
        config.tools_file.parent.mkdir(parents=True, exist_ok=True)
        old.replace(config.tools_file)
        print("tools.json перенесён в папку tools")

    if not config.tools_file.exists():
        legacy = config._read()
        tools = legacy.get("tools")
        mcp = legacy.get("mcp")
        if tools or mcp:
            config.write_tools({"tools": tools or {}, "mcp": mcp or {}})
            print("tools и mcp перенесены из config.json в tools/tools.json")
        else:
            config.tools_file.parent.mkdir(parents=True, exist_ok=True)
            example = config.root / "tools.example.json"
            if example.exists():
                shutil.copyfile(example, config.tools_file)
                print("создан tools/tools.json из tools.example.json")
            else:
                config.write_tools({"tools": {}, "mcp": {}})
                print("создан пустой tools/tools.json")

    legacy = config._read()
    if "tools" in legacy or "mcp" in legacy:
        config.save()


def run(config, llama=None, ask=input):
    """Интерактивный мастер. ask можно подменить в тестах."""
    print("\n=== Настройка Элины ===")
    prepare_files(config)
    _check_environment(config)

    _ask_model(config, llama, ask)
    _ask_voice(config, ask)
    _ask_autostart(config, ask)

    config.setup_done = True
    config.save()

    engine = Voice(config).engine or "недоступен"
    print("\nГотово:")
    print(f"  модель: {config.model or 'не выбрана'}")
    print(f"  голос: {config.voice_engine} → {engine}")
    print(f"  авто-старт LLM: {'вкл' if config.autostart_llm else 'выкл'}")
    print("  инструменты: tools/tools.json")


def _check_environment(config):
    problems = []
    if not (config.llama_path / "arch.json").exists():
        problems.append("llama.cpp не собран — запустите bash build.sh")
    if not (config.root / "models" / MODEL_NAME).is_dir():
        problems.append("модель распознавания не найдена — запустите bash build.sh")
    if problems:
        print("\nПроверка окружения:")
        for problem in problems:
            print(f"  ⚠ {problem}")


def _ask_model(config, llama, ask):
    print("\n1/3 Модель")
    models = llama.get_models() if llama else []
    if not models:
        print("  .gguf не найдены — запустите bash build.sh и повторите настройку")
        return

    for index, name in enumerate(models):
        mark = " (сейчас)" if name == config.model else ""
        print(f"  {index}) {name}{mark}")

    default = models.index(config.model) if config.model in models else 0
    raw = _question(ask, f"  номер [{default}]: ")
    if raw.isdigit() and int(raw) < len(models):
        config.model = models[int(raw)]


def _ask_voice(config, ask):
    print("\n2/3 Голос")
    available = Voice(config).available()
    piper = "да" if available["piper"] else "нет"
    espeak = "да" if available["espeak-ng"] else "нет"
    print(f"  piper: {piper} | espeak-ng: {espeak}")
    print("  0 авто | 1 piper | 2 espeak-ng | 3 выключить")

    current = config.voice_engine or "auto"
    default = next((key for key, value in ENGINES.items() if value == current), "0")
    raw = _question(ask, f"  выбор [{default}]: ") or default
    if raw in ENGINES:
        config.voice_engine = ENGINES[raw]


def _ask_autostart(config, ask):
    print("\n3/3 Авто-запуск llama-server при старте программы")
    default = "y" if config.autostart_llm else "n"
    raw = _question(ask, f"  включить? [y/n] [{default}]: ").lower()
    if raw:
        config.autostart_llm = raw in ("y", "yes", "д", "да", "1", "+")


def _question(ask, prompt):
    try:
        return ask(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""
