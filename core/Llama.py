import json
import os
import select
import subprocess
from pathlib import Path


class Llama:
    def __init__(self, config):
        self.config = config
        self.arch = self.get_arch()
        self.process = None
        self.models = []
        # ВАЖНО: это путь к директории с библиотеками, а не cwd
        self.lib_path = self.config.llama_path / self.arch["arch"] / "lib"
        self.bin_path = self.config.llama_path / self.arch["arch"] / "bin"

    def get_arch(self):
        with open(self.config.llama_path / "arch.json", "r", encoding="utf-8") as file:
            return json.load(file)

    def get_binary(self):
        binary = self.config.llama_path / self.arch["binary"]
        if not binary.exists():
            raise FileNotFoundError(f"LLAMA binary not found: {binary}")
        return binary

    def get_models(self):
        path = self.config.llama_path / "gguf"
        print("models")
        self.models = []
        for file in path.iterdir():
            if file.suffix == ".gguf":
                self.models.append(file.name)

    def start(self, threads=4):
        if self.is_running():
            raise RuntimeError("LLAMA уже запущена")

        binary = self.get_binary()
        model_path = Path(self.config.models_path).resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        env = os.environ.copy()
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            str(self.lib_path) + (":" + existing if existing else "")
        )

        command = [
            str(binary),
            "-m", str(model_path),
            "--host", self.config.host_llama,
            "--port", self.config.port_llama,
            "-t", str(threads),
            "-c", "2048",  # Меньше контекст = быстрее
            "-b", "512",  # Batch size
            "--flash-attn",  # Flash attention
            "--mlock",  # Блокировка в RAM
            "--no-mmap",  # Без mmap
        ]

        self.process = subprocess.Popen(
            command,

            cwd=str(self.bin_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        return self.process.pid

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        if self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def read_output(self, timeout=0.05):
        if self.process is None:
            return None
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            return ""
        return self.process.stdout.readline()