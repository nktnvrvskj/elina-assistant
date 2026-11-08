import json
from pathlib import Path


class Config:
    """Основные настройки: модель, голос, авто-запуск.

    Инструменты и MCP лежат отдельно, в tools.json.
    """

    SYSTEM_PROMPT = (
        "Ты краткий разговорный помощник.\n"
        "Отвечай сразу по существу.\n"
        "Не показывай ход рассуждений.\n"
        "Не пиши длинных вступлений.\n"
        "Обычно отвечай 1-3 предложениями.\n"
        "Ты ассистент тебя завут Элина"
    )

    def __init__(self):
        self.root = Path(__file__).resolve().parent.parent
        self.llama_path = self.root / "llama"
        self.config_file = self.root / "config.json"
        self.tools_file = self.root / "tools" / "tools.json"
        self.host = "127.0.0.1"
        self.port = "8080"
        self.model = None
        self.voice_engine = "auto"
        self.voice_model = None
        self.autostart_llm = False
        self.setup_done = False
        self.context_size = 8192
        self.thinking = False
        self.thinking_budget = 128
        self.memory_turns = 5
        self.load()

    @property
    def model_path(self):
        """Полный путь к выбранной .gguf модели или None."""
        if not self.model:
            return None
        return self.llama_path / "gguf" / self.model

    def load(self):
        data = self._read()
        self.model = data.get("model") or None

        voice = data.get("voice")
        voice = voice if isinstance(voice, dict) else {}
        self.voice_engine = voice.get("engine") or "auto"
        self.voice_model = voice.get("model") or None
        self.autostart_llm = bool(data.get("autostart_llm"))
        self.setup_done = bool(data.get("setup_done"))
        try:
            self.context_size = int(data.get("context_size") or 8192)
        except (TypeError, ValueError):
            self.context_size = 8192
        self.thinking = bool(data.get("thinking"))
        try:
            self.thinking_budget = int(data.get("thinking_budget", 128))
        except (TypeError, ValueError):
            self.thinking_budget = 128
        try:
            self.memory_turns = max(0, int(data.get("memory_turns", 5)))
        except (TypeError, ValueError):
            self.memory_turns = 5

    def save(self):
        """Пишет config.json; tools и mcp остаются в tools.json."""
        data = self._read()
        data.pop("tools", None)
        data.pop("mcp", None)

        voice = {"engine": self.voice_engine}
        if self.voice_model:
            voice["model"] = self.voice_model

        data.update({
            "model": self.model,
            "voice": voice,
            "autostart_llm": self.autostart_llm,
            "setup_done": self.setup_done,
            "context_size": self.context_size,
            "thinking": self.thinking,
            "thinking_budget": self.thinking_budget,
            "memory_turns": self.memory_turns,
        })
        self._write(self.config_file, data)

    def select_model(self, name):
        self.model = name
        self.save()

    def set_voice(self, engine):
        self.voice_engine = engine
        self.save()

    def set_autostart(self, enabled):
        self.autostart_llm = bool(enabled)
        self.save()

    def tools_data(self):
        """tools и mcp из tools/tools.json; старые места — запасной вариант."""
        data = self._read_json(self.tools_file)
        if not data:
            data = self._read_json(self.root / "tools.json")
        if data:
            return data
        legacy = self._read()
        return {key: legacy[key] for key in ("tools", "mcp") if key in legacy}

    def write_tools(self, data):
        self.tools_file.parent.mkdir(parents=True, exist_ok=True)
        self._write(self.tools_file, data)

    def _read(self):
        return self._read_json(self.config_file)

    @staticmethod
    def _read_json(path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _write(path, data):
        text = json.dumps(data, ensure_ascii=False, indent=2)
        Path(path).write_text(text + "\n", encoding="utf-8")
