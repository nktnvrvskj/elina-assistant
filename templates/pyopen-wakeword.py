import pyaudio
import numpy as np
from openwakeword.model import Model

# 1. В новых версиях модели скачиваются автоматически при первом запуске.
# Просто создаем модель БЕЗ аргументов (она загрузит все встроенные).
# Первый запуск может занять несколько секунд для скачивания файлов.
oww_model = Model()

# 2. Настройка микрофона
pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

print("🎙️ Слушаю через openWakeWord (новая версия)...")

try:
    while True:
        audio_chunk = stream.read(1280, exception_on_overflow=False)
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

        # Получаем предсказания
        prediction = oww_model.predict(audio_data)

        # Перебираем результаты
        for model_name, confidence in prediction.items():
            if confidence > 0.5:
                print(f"✅ АКТИВАЦИЯ! Модель: {model_name} ({confidence:.2f})")
                # oww_model.reset() # Раскомментируй, если нужно сбрасывать состояние

except KeyboardInterrupt:
    print("\n⏹️ Остановлено")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()