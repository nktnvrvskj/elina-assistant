import json
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path


class Llama:
    """Запуск, остановка и лог llama-server."""

    # Флаги llama.cpp меняются между версиями: сначала новые, потом старые,
    # в конце самый простой набор.
    FLAG_VARIANTS = (
        ("-fa", "on", "--load-mode", "mmap+mlock"),
        ("--flash-attn", "--mlock", "--no-mmap"),
        (),
    )

    def __init__(self, config):
        self.config = config
        self.process = None
        self.models = []
        self._log = deque(maxlen=500)
        self._unread = queue.Queue()

    def _arch(self):
        path = self.config.llama_path / "arch.json"
        if not path.exists():
            hint = "build.ps1" if os.name == "nt" else "bash build.sh"
            raise FileNotFoundError(f"llama не собран — запустите {hint}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_binary(self):
        binary = self.config.llama_path / self._arch()["binary"]
        if os.name == "nt" and binary.suffix.lower() != ".exe":
            candidate = binary.with_name(binary.name + ".exe")
            if candidate.exists():
                binary = candidate
        if not binary.exists():
            raise FileNotFoundError(f"нет llama-server: {binary}")
        return binary

    def get_models(self):
        """Список .gguf в llama/gguf."""
        folder = self.config.llama_path / "gguf"
        self.models = (
            sorted(file.name for file in folder.glob("*.gguf"))
            if folder.is_dir() else []
        )
        return self.models

    def read_output(self):
        try:
            return self._unread.get_nowait()
        except queue.Empty:
            return ""

    def last_error(self, lines=15):
        return " | ".join(list(self._log)[-lines:])

    def start(self, threads=None):
        if self.is_running():
            raise RuntimeError("llama-server уже запущен")
        if threads is None:
            threads = min(os.cpu_count() or 4, 8)

        model_path = self.config.model_path
        if model_path is None:
            raise RuntimeError("модель не выбрана — нажмите 1 в меню")
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"нет файла модели: {model_path}")

        arch = self._arch()
        binary = self.get_binary()
        lib_path = self.config.llama_path / arch["arch"] / "lib"
        bin_path = self.config.llama_path / arch["arch"] / "bin"

        env = os.environ.copy()
        if os.name == "nt":
            env["PATH"] = str(bin_path) + os.pathsep + env.get("PATH", "")
        else:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = str(lib_path) + (":" + existing if existing else "")

        base = [
            str(binary),
            "-m", str(model_path),
            "--host", self.config.host,
            "--port", self.config.port,
            "-t", str(threads),
            "-c", str(self._context_size()),
            "-b", "512",
            "--swa-full",
        ]
        budget = int(getattr(self.config, "thinking_budget", 128) or 0)
        if getattr(self.config, "thinking", False) and budget != 0:
            base += ["--reasoning", "on", "--reasoning-budget", str(budget)]
            if budget > 0:
                base += ["--reasoning-budget-message",
                         "Хватит рассуждать, ответь кратко."]
        else:
            base += ["--reasoning", "off"]

        if getattr(self.config, "thinking", False) and budget > 0:
            self._warn_slow_thinking(budget)

        errors = []
        for extra in self.FLAG_VARIANTS:
            self.process = subprocess.Popen(
                base + list(extra),
                cwd=str(bin_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
            self._log.clear()
            self._unread = queue.Queue()
            threading.Thread(target=self._pump, daemon=True).start()

            if self._wait_ready(timeout=120):
                return self.process.pid

            errors.append(self.last_error(8))
            self.stop()

        raise RuntimeError(
            "llama-server не запустился.\n" + (errors[-1] or "нет вывода")
        )

    def _pump(self):
        """Читаем вывод в фоне, иначе pipe переполняется и сервер встаёт."""
        for line in self.process.stdout:
            text = line.rstrip()
            self._log.append(text)
            self._unread.put(text)

    def _health(self):
        try:
            url = f"http://{self.config.host}:{self.config.port}/health"
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _wait_ready(self, timeout=120):
        """Ждём /health, пока сервер грузит модель."""
        deadline = time.monotonic() + timeout
        announced = False
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return False
            if self._health():
                return True
            if not announced and time.monotonic() > deadline - timeout + 3:
                print("запускаю llama-server, жду загрузки модели…")
                announced = True
            time.sleep(0.5)
        return False

    @staticmethod
    def _warn_slow_thinking(budget):
        import platform

        machine = platform.machine().lower()
        if machine.startswith(("aarch64", "arm", "armv7")):
            seconds = budget / 10
            print(f"thinking включён: на ARM при ~10 ток/с бюджет "
                  f"{budget} токенов — это ~{seconds:.0f} с на обычный вопрос "
                  f"(вызовы инструментов идут без рассуждений)")

    def _context_size(self):
        try:
            return max(512, int(self.config.context_size))
        except (TypeError, ValueError):
            return 8192

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def is_ready(self):
        """Наш сервер жив или уже запущенный кем-то другим отвечает на /health."""
        return self.is_running() or self._health()

    def stop(self):
        if self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process is not None and self.process.stdout:
            try:
                self.process.stdout.close()
            except Exception:
                pass
        self.process = None
