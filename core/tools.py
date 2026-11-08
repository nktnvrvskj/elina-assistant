"""Инструменты, которые ИИ может вызывать.

Встроенные — погода, время, запуск программ и ссылок.
Свои — файл в tools/ с декоратором @tool или команда в tools.json.
MCP-серверы — секция "mcp" в tools.json.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable

from core.mcp import McpError, McpServer, load_servers

WEEKDAYS = ("понедельник", "вторник", "среда", "четверг",
            "пятница", "суббота", "воскресенье")
USER_AGENT = "elina-assistant/0.1"


class Tool:
    """Один инструмент: имя, описание, параметры и функция."""

    def __init__(self, name: str, description: str,
                 params: tuple[str, ...], func: Callable[[dict], str],
                 source: str = "встроенный") -> None:
        self.name = name
        self.description = description
        self.params = params
        self.func = func
        self.source = source

    def call(self, arguments: dict | None) -> str:
        arguments = dict(arguments or {})
        if self.params:
            # Выкидываем аргументы, которых нет в описании инструмента.
            arguments = {key: value for key, value in arguments.items()
                         if key in self.params}
        return self.func(arguments)

    def signature(self) -> str:
        if not self.params:
            return f"{self.name}()"
        return f"{self.name}({', '.join(self.params)})"

    def prompt_line(self) -> str:
        return f"- {self.signature()} — {self.description}"


_USER_TOOLS: dict[str, Tool] = {}


def _coerce(value, annotation):
    """Модель присылает строки — приводим к типу из аннотации."""
    if value is None or annotation is inspect.Parameter.empty:
        return value
    try:
        if annotation is bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "да", "on")
        if annotation is int:
            return int(value)
        if annotation is float:
            return float(value)
        if annotation is str:
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def _from_function(name: str, description: str, func) -> Tool:
    signature = inspect.signature(func)
    params = tuple(signature.parameters)

    def call(arguments: dict) -> str:
        kwargs = {}
        for param in params:
            if param in arguments:
                kwargs[param] = _coerce(
                    arguments[param],
                    signature.parameters[param].annotation,
                )
        return str(func(**kwargs))

    return Tool(name, description, params, call, source="tools/")


def tool(name: str | None = None, description: str | None = None):
    """Декоратор для своих инструментов.

    @tool("battery", "заряд батареи")
    def battery() -> str: ...
    """
    def register(func):
        tool_name = name or func.__name__
        text = description
        if not text:
            doc = (func.__doc__ or "").strip()
            text = doc.splitlines()[0] if doc else tool_name
        _USER_TOOLS[tool_name] = _from_function(tool_name, text, func)
        return func

    return register


def load_user_tools(folder: Path) -> tuple[list[Tool], dict[str, str]]:
    """Импортирует все tools/*.py и собирает из них инструменты."""
    errors: dict[str, str] = {}
    if not folder.is_dir():
        return [], errors
    # Чтобы `from core.tools import tool` работал и при запуске -m core.tools.
    sys.modules.setdefault("core.tools", sys.modules[__name__])
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"elina_tools_{path.stem}"
        if module_name in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as error:
            errors[path.name] = f"{type(error).__name__}: {error}"
    return list(_USER_TOOLS.values()), errors


def _read_json(path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_json(config) -> dict:
    """tools и mcp из tools/tools.json; старые места — запасной вариант."""
    loader = getattr(config, "tools_data", None)
    if callable(loader):
        return loader()

    tools_file = getattr(config, "tools_file", None)
    if tools_file:
        data = _read_json(tools_file)
        if data:
            return data
        old = Path(tools_file).parent.parent / "tools.json"
        data = _read_json(old)
        if data:
            return data

    config_file = getattr(config, "config_file", None)
    if config_file:
        data = _read_json(config_file)
        return {key: data[key] for key in ("tools", "mcp") if key in data}
    return {}


def _get_weather(city: str) -> str:
    city = (city or "").strip() or "Москва"
    # Проценты и запятые в format= оставляем как есть, кириллицу кодируем.
    template = "%l: %c %t, ощущается %f, влажность %h, ветер %w"
    url = ("https://wttr.in/" + urllib.parse.quote(city)
           + "?format=" + urllib.parse.quote(template, safe="%:+,.")
           + "&lang=ru")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8", "replace").strip()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return f"не смог узнать погоду: {error}"

    if text.startswith("<"):
        return "сервис погоды сейчас недоступен, попробуйте позже"
    return " ".join(text.split())


def _get_time() -> str:
    now = time.localtime()
    return (f"{now.tm_mday:02d}.{now.tm_mon:02d}.{now.tm_year}, "
            f"{now.tm_hour:02d}:{now.tm_min:02d} "
            f"({WEEKDAYS[now.tm_wday]})")


def _open_url(url: str) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url if url else ""
    if not url:
        return "не указан адрес"
    if not webbrowser.open(url):
        return f"не удалось открыть {url}"
    return f"открыл {url}"


class Tools:
    """Реестр инструментов: встроенные, свои (tools/, config) и MCP."""

    def __init__(self, config=None) -> None:
        self.config = config
        data = _load_json(config)
        self.settings = data.get("tools") if isinstance(data.get("tools"), dict) else {}
        self._mcp: dict[str, McpServer] = load_servers(data.get("mcp"))
        self._mcp_tools: list[Tool] | None = None
        self._mcp_errors: dict[str, str] = {}
        self._builtin = self._make_builtin()
        self._user, self._tool_errors = load_user_tools(self.tools_dir())
        self._commands = self._make_commands()

    def tools_dir(self) -> Path:
        """Папка со своими инструментами (по умолчанию tools/)."""
        custom = self.settings.get("dir")
        if custom:
            return Path(str(custom)).expanduser()
        return Path(__file__).resolve().parent.parent / "tools"

    def _make_commands(self) -> list[Tool]:
        """Инструменты-команды из tools.json (tools.commands)."""
        commands = self.settings.get("commands") or {}
        result: list[Tool] = []
        for name, options in commands.items():
            if isinstance(options, str):
                options = {"run": options}
            if not isinstance(options, dict):
                continue
            command = options.get("run")
            if not command:
                continue
            description = options.get("description") or f"команда «{name}»"
            result.append(Tool(
                name, description, (),
                (lambda args, cmd=command: self._run_command(cmd)),
                source="tools.json",
            ))
        return result

    def _run_command(self, command: str) -> str:
        try:
            finished = subprocess.run(command, shell=True, capture_output=True,
                                      text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return "команда не успела выполниться за 60 секунд"
        output = (finished.stdout or "").strip() or (finished.stderr or "").strip()
        if finished.returncode != 0 and not output:
            return f"команда завершилась с кодом {finished.returncode}"
        return (output or "готово")[:800]

    def _make_builtin(self) -> list[Tool]:
        apps = self.settings.get("apps") or {}
        apps_hint = ("Программы: " + ", ".join(apps)) if apps else ""

        return [
            Tool("get_weather", "погода в городе (температура, ветер, влажность)",
                 ("city",), lambda a: _get_weather(a.get("city", ""))),
            Tool("get_time", "текущие дата, время и день недели",
                 (), lambda a: _get_time()),
            Tool("open_app", "запустить программу из списка" +
                 (f" ({apps_hint})" if apps_hint else ""),
                 ("name",), lambda a: self._open_app(a.get("name", ""))),
            Tool("open_url", "открыть сайт в браузере",
                 ("url",), lambda a: _open_url(a.get("url", ""))),
        ]

    def _open_app(self, name: str) -> str:
        apps = self.settings.get("apps") or {}
        name = str(name or "").strip()
        if not apps:
            return ("список программ не настроен: добавьте tools.apps в tools.json "
                    '(например "браузер": "firefox")')
        command = apps.get(name)
        if command is None:
            for key, value in apps.items():
                if key.lower() == name.lower():
                    command = value
                    break
        if command is None:
            return f"нет программы «{name}». Доступны: {', '.join(apps)}"
        try:
            subprocess.Popen(
                shlex.split(command),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, ValueError) as error:
            return f"не смог запустить «{name}»: {error}"
        return f"запустил «{name}»"

    def _ensure_mcp(self) -> list[Tool]:
        if self._mcp_tools is not None:
            return self._mcp_tools

        tools: list[Tool] = []
        for name, server in self._mcp.items():
            try:
                server.start()
                listed = server.list_tools()
            except (McpError, OSError) as error:
                self._mcp_errors[name] = str(error)
                continue
            for item in listed:
                properties = ((item.get("inputSchema") or {}).get("properties") or {})
                tools.append(Tool(
                    f"{name}.{item['name']}",
                    item.get("description") or f"инструмент {name}",
                    tuple(properties),
                    (lambda args, srv=server, tool=item["name"]:
                     srv.call_tool(tool, args)),
                ))
        self._mcp_tools = tools
        return tools

    def specs(self) -> list[Tool]:
        """Все доступные инструменты (MCP-серверы стартуют при первом вызове)."""
        return self._builtin + self._user + self._commands + self._ensure_mcp()

    def names(self) -> set[str]:
        return {tool.name for tool in self.specs()}

    def call(self, name: str, arguments: dict | None = None) -> str:
        for tool in self.specs():
            if tool.name == name:
                try:
                    return str(tool.call(arguments))
                except McpError as error:
                    return f"ошибка инструмента: {error}"
                except Exception as error:
                    return f"ошибка инструмента: {error}"
        return f"неизвестный инструмент «{name}»"

    def hint(self) -> str:
        """Дополнительный текст для системного промпта."""
        lines = [tool.prompt_line() for tool in self.specs()]
        if self._mcp_errors:
            lines.append("Недоступные MCP-серверы: " + ", ".join(
                f"{name} ({error})" for name, error in self._mcp_errors.items()))
        if self._tool_errors:
            lines.append("Ошибки в своих инструментах: " + ", ".join(
                f"{name} ({error})" for name, error in self._tool_errors.items()))
        return "\n".join(lines)

    def close(self) -> None:
        for server in self._mcp.values():
            server.close()


def main(argv=None) -> int:
    """Проверка инструментов без llama: python -m core.tools --list"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m core.tools",
        description="Инструменты ИИ: список и ручной вызов.")
    parser.add_argument("--config", help="путь к config.json")
    parser.add_argument("--tools", help="путь к tools.json")
    parser.add_argument("--list", action="store_true", help="показать инструменты")
    parser.add_argument("--call", nargs=2, metavar=("TOOL", "ARGS_JSON"),
                        help='вызвать инструмент, аргументы — JSON: '
                             '--call get_weather \'{"city": "Москва"}\'')
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent

    class LocalConfig:
        config_file = Path(args.config) if args.config else root / "config.json"
        tools_file = (Path(args.tools) if args.tools
                      else root / "tools" / "tools.json")

    tools = Tools(LocalConfig())
    try:
        if args.call:
            name, raw = args.call
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                raise SystemExit('аргументы — JSON, например: \'{"city": "Москва"}\'')
            print(tools.call(name, payload))
        else:
            for tool in tools.specs():
                print(f"{tool.prompt_line()}   [{tool.source}]")
            errors = dict(getattr(tools, "_tool_errors", {}))
            errors.update(getattr(tools, "_mcp_errors", {}))
            for name, error in errors.items():
                print(f"⚠  {name}: {error}", file=sys.stderr)
    finally:
        tools.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
