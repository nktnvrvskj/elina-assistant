"""Минимальный клиент MCP (Model Context Protocol) без внешних зависимостей.

Транспорт stdio: сервер — дочерний процесс, обмен идёт JSON-RPC 2.0, по
одному JSON на строку.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque

PROTOCOL_VERSION = "2024-11-05"


class McpError(RuntimeError):
    pass


class McpServer:
    """Один MCP-сервер, запущенный как дочерний процесс."""

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 env: dict | None = None, cwd: str | None = None,
                 timeout: float = 30.0) -> None:
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout

        self._proc: subprocess.Popen | None = None
        self._pending: dict[int, queue.Queue] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._stderr = deque(maxlen=20)
        self._tools: list[dict] | None = None
        self.server_info: dict = {}

    def start(self) -> "McpServer":
        if self._proc is not None:
            return self
        env = os.environ.copy()
        env.update(self.env)
        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                bufsize=1, env=env, cwd=self.cwd,
            )
        except FileNotFoundError as error:
            raise McpError(
                f"MCP «{self.name}»: команда не найдена — {self.command}"
            ) from error

        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._initialize()
        return self

    def _pump_stdout(self) -> None:
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_id = message.get("id")
            if msg_id is None:
                continue
            waiter = self._pending.get(msg_id)
            if waiter is not None:
                waiter.put(message)

    def _pump_stderr(self) -> None:
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())

    def _write(self, message: dict) -> None:
        if self._proc is None or self._proc.poll() is not None:
            tail = " | ".join(self._stderr) or "процесс не запущен"
            raise McpError(f"MCP «{self.name}» не работает: {tail}")
        try:
            self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            tail = " | ".join(self._stderr)
            raise McpError(f"MCP «{self.name}» оборвался: {tail or error}") from error

    def _request(self, method: str, params: dict | None = None,
                 timeout: float | None = None) -> dict:
        if self._proc is None:
            self.start()
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            waiter: queue.Queue = queue.Queue(maxsize=1)
            self._pending[request_id] = waiter

        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._write(message)

        try:
            response = waiter.get(timeout=timeout or self.timeout)
        except queue.Empty:
            raise McpError(f"MCP «{self.name}»: нет ответа на {method} "
                           f"за {timeout or self.timeout:.0f} с") from None
        finally:
            self._pending.pop(request_id, None)

        if "error" in response:
            error = response["error"]
            raise McpError(f"MCP «{self.name}»: {error.get('message', error)}")
        return response.get("result") or {}

    def notify(self, method: str, params: dict | None = None) -> None:
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._write(message)

    def _initialize(self) -> None:
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "elina-assistant", "version": "0.1"},
        })
        self.server_info = result.get("serverInfo", {})
        self.notify("notifications/initialized")

    def list_tools(self, refresh: bool = False) -> list[dict]:
        """Список инструментов сервера (кэшируется)."""
        if self._tools is None or refresh:
            self._tools = self._request("tools/list").get("tools", [])
        return self._tools

    def call_tool(self, name: str, arguments: dict | None = None,
                  timeout: float | None = None) -> str:
        """Вызывает инструмент и возвращает текстовый результат."""
        result = self._request("tools/call",
                               {"name": name, "arguments": arguments or {}},
                               timeout=timeout)
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(f"[{item.get('type', 'объект')}]")
        text = "\n".join(part for part in parts if part).strip()
        if result.get("isError"):
            return f"ошибка: {text or 'инструмент вернул ошибку'}"
        return text or "готово"

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        except Exception:
            pass


def load_servers(config: dict | None) -> dict[str, McpServer]:
    """Создаёт (но не запускает) серверы из секции config["mcp"]."""
    servers: dict[str, McpServer] = {}
    for name, options in (config or {}).items():
        if not isinstance(options, dict) or options.get("enabled") is False:
            continue
        command = options.get("command")
        if not command:
            print(f"⚠  MCP «{name}»: не задан command", file=sys.stderr)
            continue
        servers[name] = McpServer(
            name,
            command,
            args=options.get("args"),
            env=options.get("env"),
            cwd=options.get("cwd"),
            timeout=float(options.get("timeout", 30)),
        )
    return servers
