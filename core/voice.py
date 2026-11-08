"""Голос помощника: слушает «Элину» и озвучивает ответы.

Озвучка — Piper (нейроголос, офлайн) или espeak-ng (лёгкий, системный).
Движок выбирается в config.json (voice.engine) или в меню, пункт 7.

Проверка из терминала:

    python -m core.voice --list
    python -m core.voice --say "Привет, я Элина"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

DEFAULT_PIPER_MODEL = "models/piper/ru_RU-irina-medium.onnx"


class Voice:
    """Голосовой ввод (Vosk) и озвучка ответов (Piper / espeak-ng)."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config.root)
        self.problem = ""
        self._wake = None
        self._piper_voice = None
        self._warned = False
        self.engine = self._detect()

    def _model_path(self) -> Path:
        custom = getattr(self.config, "voice_model", None)
        if custom:
            path = Path(custom).expanduser()
            return path if path.is_absolute() else self.root / path
        return self.root / DEFAULT_PIPER_MODEL

    def _piper_available(self) -> bool:
        return (importlib.util.find_spec("piper") is not None
                and self._model_path().exists())

    @staticmethod
    def _espeak_bin():
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def available(self) -> dict:
        """Какие движки установлены: {"piper": bool, "espeak-ng": bool}."""
        return {
            "piper": self._piper_available(),
            "espeak-ng": self._espeak_bin() is not None,
        }

    def _detect(self):
        want = (getattr(self.config, "voice_engine", None) or "auto").lower()
        if want == "none":
            return None

        piper_ok = self._piper_available()
        espeak_ok = self._espeak_bin() is not None

        if want == "piper":
            if piper_ok:
                return "piper"
            if importlib.util.find_spec("piper") is None:
                self.problem = "piper не установлен (pip install piper-tts)"
            else:
                self.problem = f"нет модели piper: {self._model_path()}"
            return None
        if want == "espeak-ng":
            if espeak_ok:
                return "espeak-ng"
            self.problem = "espeak-ng не найден в PATH"
            return None

        if piper_ok:
            return "piper"
        if espeak_ok:
            return "espeak-ng"
        self.problem = "нет TTS: поставьте piper-tts или espeak-ng"
        return None

    def listen_command(self) -> str:
        """Ждёт «Элину» и возвращает фразу после неё."""
        if self._wake is None:
            from core.wakeword import WakeWord
            self._wake = WakeWord()
        return self._wake.listen_command()

    def say(self, text: str) -> bool:
        """Озвучивает текст. False — если движок недоступен или не вышло."""
        text = _clean_for_speech(text)
        if not text:
            return False
        if self.engine is None:
            self._warn()
            return False

        handle, name = tempfile.mkstemp(prefix="elina-", suffix=".wav")
        os.close(handle)
        path = Path(name)
        try:
            self._synthesize(text, path)
            if self._play(path):
                return True
            print("голос: не удалось воспроизвести звук", file=sys.stderr)
        except Exception as error:
            print(f"голос: ошибка озвучки: {error}", file=sys.stderr)
        finally:
            path.unlink(missing_ok=True)
        return False

    def _warn(self):
        if self.problem and not self._warned:
            print(f"голос: {self.problem}", file=sys.stderr)
            self._warned = True

    def _synthesize(self, text: str, path: Path):
        if self.engine == "piper":
            if self._piper_voice is None:
                from piper import PiperVoice
                self._piper_voice = PiperVoice.load(str(self._model_path()))
            with wave.open(str(path), "wb") as wav:
                self._piper_voice.synthesize_wav(text, wav)
            return

        binary = self._espeak_bin()
        subprocess.run(
            [binary, "-v", "ru", "-s", "165", "-w", str(path), text],
            check=True, capture_output=True, timeout=120,
        )

    def _play(self, path: Path) -> bool:
        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME)
                return True
            except Exception:
                pass

        try:
            _play_sounddevice(path)
            return True
        except Exception:
            pass

        for command in (["paplay"], ["pw-play"], ["aplay", "-q"]):
            if not shutil.which(command[0]):
                continue
            try:
                done = subprocess.run(command + [str(path)],
                                      capture_output=True, timeout=120)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if done.returncode == 0:
                return True
        return False

    def close(self):
        self._wake = None
        self._piper_voice = None


def _clean_for_speech(text: str) -> str:
    """Убирает разметку и ссылки, чтобы голос не читал мусор."""
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+", " ссылка ", text)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


def _play_sounddevice(path: Path):
    import sounddevice as sd

    with wave.open(str(path), "rb") as wav:
        stream = sd.RawOutputStream(
            samplerate=wav.getframerate(),
            channels=wav.getnchannels(),
            dtype="int16",
        )
        stream.start()
        try:
            while True:
                chunk = wav.readframes(4096)
                if not chunk:
                    break
                stream.write(chunk)
        finally:
            stream.stop()
            stream.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m core.voice",
        description="Озвучка ответов: piper или espeak-ng.")
    parser.add_argument("--say", metavar="TEXT", help="озвучить текст")
    parser.add_argument("--engine", choices=("auto", "piper", "espeak-ng", "none"),
                        help="движок озвучки")
    parser.add_argument("--model", help="путь к модели piper (.onnx)")
    parser.add_argument("--list", action="store_true", help="показать движки")
    args = parser.parse_args(argv)

    from core.config import Config

    config = Config()
    if args.engine:
        config.voice_engine = args.engine
    if args.model:
        config.voice_model = args.model

    voice = Voice(config)
    if args.list:
        print("piper:", "да" if voice._piper_available() else "нет",
              f"({voice._model_path()})")
        print("espeak-ng:", voice._espeak_bin() or "нет")
        print("выбран:", voice.engine or "нет", voice.problem)
        return 0
    if args.say:
        return 0 if voice.say(args.say) else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
