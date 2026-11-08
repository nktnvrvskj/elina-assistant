"""
core/WakeWord.py — офлайн-детектор кодового слова «Элина».

Работает на Vosk (Apache-2.0): компактная русская модель (~45 МБ) целиком
крутится на CPU, интернет нужен только один раз — чтобы её скачать.
Никаких ключей, аккаунтов и подписок.

Использование в коде:

    from core.WakeWord import WakeWord

    wake = WakeWord()
    if wake.wait(timeout=30):          # ждём «Элину»
        print("услышал")
    text = wake.listen_command()       # «Элина, какая погода?» -> «какая погода»

Запуск для проверки без микрофона и без вашего кода:

    python -m core.WakeWord                  # слушать микрофон, печатать ЭЛИНА
    python -m core.WakeWord --command        # печатать распознанную команду
    python -m core.WakeWord --wav file.wav   # прогнать аудиофайл
    python -m core.WakeWord --list-devices   # список микрофонов

Модель и зависимости ставит build.sh (см. README).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
import wave
from array import array
from difflib import SequenceMatcher
from pathlib import Path

SAMPLE_RATE = 16000
BLOCK = 512                      # 32 мс: низкая задержка, нагрузка всё ещё копеечная
MODEL_NAME = "vosk-model-small-ru-0.22"
DEFAULT_ALIASES = ("элена", "елина", "элино", "элины")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


# ===========================================================================
#  Модель
# ===========================================================================
def _looks_like_model(path: Path) -> bool:
    return path.is_dir() and (path / "am").is_dir() and (path / "conf").is_dir()


def find_model() -> Path:
    """Возвращает папку с моделью Vosk, которую ставит build.sh."""
    model = MODELS_DIR / MODEL_NAME
    if _looks_like_model(model):
        return model
    raise SystemExit(
        "Модель распознавания речи не найдена.\n"
        "Запустите установку:  bash build.sh\n"
        f"Ожидается папка: {model}"
    )


# ===========================================================================
#  Ресемплинг, если микрофон не умеет 16 кГц
# ===========================================================================
class _Resampler:
    """Линейная интерполяция PCM int16 (без numpy)."""

    def __init__(self, in_rate: int, out_rate: int = SAMPLE_RATE) -> None:
        self.step = in_rate / out_rate
        self.pos = 0.0
        self.buf = array("h")

    def process(self, chunk: bytes) -> bytes:
        self.buf.extend(array("h", chunk))
        out = array("h")
        n = len(self.buf)
        pos = self.pos
        while True:
            i = int(pos)
            if i + 1 >= n:
                break
            frac = pos - i
            a = self.buf[i]
            out.append(int(a + (self.buf[i + 1] - a) * frac))
            pos += self.step
        cut = int(pos)
        if cut:
            del self.buf[:cut]
            pos -= cut
        self.pos = pos
        return out.tobytes()


# ===========================================================================
#  Сопоставление текста со словом
# ===========================================================================
def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def _similarity(candidate: str, alias: str) -> float:
    if candidate == alias:
        return 1.0
    if len(candidate) > len(alias) and candidate.startswith(alias) \
            and len(candidate) - len(alias) <= 3:
        return 0.95                                   # элина -> элину/элиной
    if len(candidate) >= 4 and alias.startswith(candidate):
        return 0.90                                   # недослышали: «элин»
    return SequenceMatcher(None, candidate, alias).ratio()


def match_score(text: str, aliases) -> tuple[float, str]:
    """Лучшее совпадение по словам, парам и тройкам («э ли на» -> «элина»)."""
    words = normalize(text).split()
    if not words:
        return 0.0, ""
    candidates = list(words)
    for size in (2, 3):
        for i in range(len(words) - size + 1):
            candidates.append("".join(words[i:i + size]))

    best, best_c = 0.0, ""
    for cand in candidates:
        if len(cand) < 3:
            continue
        for alias in aliases:
            score = _similarity(cand, alias)
            if score > best:
                best, best_c = score, cand
    return best, best_c


# ===========================================================================
#  Детектор
# ===========================================================================
class WakeWord:
    """Слушает микрофон и срабатывает на слово «Элина»."""

    def __init__(self, model_dir: str | Path | None = None,
                 word: str = "элина",
                 aliases: tuple[str, ...] = DEFAULT_ALIASES,
                 threshold: float = 0.82,
                 cooldown: float = 2.0,
                 device: str | int | None = None,
                 verbose: bool = False) -> None:
        self.word = normalize(word)
        self.aliases = tuple(a for a in (normalize(x) for x in (word, *aliases)) if a)
        self.threshold = threshold
        self.cooldown = cooldown
        self.device = device
        self.verbose = verbose
        self.hits = 0
        self._last_hit = -1e9

        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self._model = Model(str(model_dir or find_model()))
        self._rec = KaldiRecognizer(self._model, SAMPLE_RATE)

    # -- микрофон ----------------------------------------------------------
    @contextlib.contextmanager
    def _mic(self):
        """Открывает микрофон и отдаёт бесконечный поток блоков PCM 16 кГц."""
        import sounddevice as sd

        rate, blocksize, resampler = SAMPLE_RATE, BLOCK, None
        kwargs: dict = dict(dtype="int16", channels=1)
        if self.device is not None:
            kwargs["device"] = self.device

        try:
            stream = sd.RawInputStream(samplerate=rate, blocksize=blocksize, **kwargs)
            stream.start()
        except sd.PortAudioError:
            info = sd.query_devices(self.device, "input")
            rate = int(info["default_samplerate"])
            blocksize = BLOCK * rate // SAMPLE_RATE
            resampler = _Resampler(rate, SAMPLE_RATE)
            if self.verbose:
                print(f"ℹ  микрофон на {rate} Гц, пересчитываю в 16 кГц", file=sys.stderr)
            stream = sd.RawInputStream(samplerate=rate, blocksize=blocksize, **kwargs)
            stream.start()

        try:
            def blocks():
                while True:
                    data, overflowed = stream.read(blocksize)
                    if overflowed:
                        continue
                    yield resampler.process(bytes(data)) if resampler else bytes(data)

            yield blocks()
        finally:
            stream.stop()
            stream.close()

    # -- распознавание -----------------------------------------------------
    def _feed(self, pcm: bytes) -> bool:
        """Скормить блок. True — только что услышали слово."""
        if self._rec.AcceptWaveform(pcm):
            text = json.loads(self._rec.Result()).get("text", "")
        else:
            text = json.loads(self._rec.PartialResult()).get("partial", "")

        if self.verbose and text:
            sys.stderr.write(f"\r\033[K   слышу: {text}")
            sys.stderr.flush()

        score, cand = match_score(text, self.aliases)
        if score < self.threshold:
            return False

        now = time.monotonic()
        if now - self._last_hit < self.cooldown:
            return False
        self._last_hit = now
        self.hits += 1
        if self.verbose:
            sys.stderr.write(f"\r\033[K   ✓ {score:.2f} «{cand}»\n")
        return True

    def _text_after_pause(self, blocks, limit: float) -> str:
        """Копит фразу до паузы (или limit секунд) и возвращает текст."""
        deadline = time.monotonic() + limit
        text = ""
        for block in blocks:
            if time.monotonic() > deadline:
                break
            if self._rec.AcceptWaveform(block):
                text = json.loads(self._rec.Result()).get("text", "")
                break
            text = json.loads(self._rec.PartialResult()).get("partial", "")
        return text.strip()

    def _strip_word(self, text: str) -> str:
        """Убирает «элина» из начала распознанной фразы."""
        words = text.split()
        for i, token in enumerate(words[:3]):
            if match_score(token, self.aliases)[0] >= self.threshold:
                del words[i]
                break
        return " ".join(words)

    # -- публичный интерфейс ----------------------------------------------
    def wait(self, timeout: float | None = None) -> bool:
        """Блокирующе ждёт слово. False — если вышел timeout."""
        started = time.monotonic()
        with self._mic() as blocks:
            for block in blocks:
                if timeout is not None and time.monotonic() - started > timeout:
                    return False
                if self._feed(block):
                    return True
        return False

    def listen(self):
        """Генератор: отдаёт управление каждый раз, когда услышал слово."""
        with self._mic() as blocks:
            for block in blocks:
                if self._feed(block):
                    yield

    def listen_command(self, max_seconds: float = 10.0,
                       timeout: float | None = None) -> str:
        """Ждёт «Элину», затем возвращает сказанное после неё.

        Понимает оба варианта: «Элина, какая погода?» и «Элина» <пауза> «какая
        погода». Возвращает пустую строку, если после слова ничего не сказали.
        """
        started = time.monotonic()
        with self._mic() as blocks:
            for block in blocks:
                if timeout is not None and time.monotonic() - started > timeout:
                    return ""
                if self._feed(block):
                    break
            else:
                return ""

            # фраза целиком: «элина какая погода» -> «какая погода»
            command = self._strip_word(self._text_after_pause(blocks, max_seconds))
            if not command:
                # слово сказали отдельно — ждём отдельную фразу-команду
                self._rec.Reset()
                command = self._text_after_pause(blocks, max_seconds)
            self._rec.Reset()
            return command.strip()


# ===========================================================================
#  Проверка из командной строки
# ===========================================================================
def _read_wav(path: Path):
    """WAV 16 кГц/моно/16 бит блоками; другое — через ffmpeg."""
    import shutil
    import subprocess
    import tempfile

    tmp = None
    try:
        with wave.open(str(path), "rb") as wf:
            if not (wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1
                    and wf.getsampwidth() == 2):
                raise wave.Error("нужна конвертация")
    except (wave.Error, EOFError):
        if not shutil.which("ffmpeg"):
            raise SystemExit(f"{path.name}: нужен WAV 16 кГц/моно или ffmpeg")
        fd, name = tempfile.mkstemp(suffix=".wav")
        import os
        os.close(fd)
        tmp = Path(name)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                        str(tmp)], check=True)
        path = tmp

    try:
        with wave.open(str(path), "rb") as wf:
            while True:
                data = wf.readframes(BLOCK)
                if not data:
                    return
                yield data
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m core.WakeWord",
        description="Офлайн-детектор слова «Элина» (Vosk).")
    parser.add_argument("--device", help="микрофон: номер или часть названия")
    parser.add_argument("--list-devices", action="store_true", help="список микрофонов")
    parser.add_argument("--command", action="store_true", help="печатать фразу после слова")
    parser.add_argument("--wav", metavar="FILE", help="проверить на файле")
    parser.add_argument("--bench", metavar="FILE",
                        help="замерить скорость на файле (для Raspberry Pi)")
    parser.add_argument("--word", default="элина", help="другое кодовое слово")
    parser.add_argument("--threshold", type=float, default=0.82, help="порог 0…1")
    parser.add_argument("-v", "--verbose", action="store_true", help="показывать, что слышно")
    args = parser.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"[{i}] {dev['name']} ({int(dev['default_samplerate'])} Гц)")
        return 0

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    wake = WakeWord(word=args.word, threshold=args.threshold,
                    device=device, verbose=args.verbose)

    if args.wav or args.bench:
        src = args.wav or args.bench
        total = 0.0
        hits = []
        started = time.perf_counter()
        for block in _read_wav(Path(src)):
            total += len(block) / 2 / SAMPLE_RATE
            if wake._feed(block):
                hits.append(total)
        spent = time.perf_counter() - started
        for at in hits:
            print(f"ЭЛИНА (на {at:.2f} c)")
        if not hits:
            print("слово не найдено")
        if args.bench:
            print(f"аудио {total:.2f} c, обработка {spent:.2f} c "
                  f"(x{total / spent:.1f} реального времени)")
        return 0

    print("🎙  Скажите «Элина»… Ctrl+C — выход")
    try:
        if args.command:
            while True:
                text = wake.listen_command()
                print(f"вы: {text or '—'}")
        else:
            for _ in wake.listen():
                print("ЭЛИНА", flush=True)
    except KeyboardInterrupt:
        print("\nвыход")
    return 0


if __name__ == "__main__":
    sys.exit(main())
