"""Пример своего инструмента.

Скопируйте файл, переименуйте и добавьте функции с @tool — ИИ увидит их
после перезапуска. Описание берётся из первой строки докстринга.
"""

import shutil

from core.tools import tool


@tool("disk_free", "сколько свободного места на диске")
def disk_free() -> str:
    """Свободное место на разделе с проектом."""
    usage = shutil.disk_usage("/")
    return f"свободно {usage.free / 1e9:.1f} ГБ из {usage.total / 1e9:.1f} ГБ"


# Инструмент с параметром: тип берётся из аннотации.
#
# @tool("set_volume", "установить громкость 0-100")
# def set_volume(level: int) -> str:
#     subprocess.run(["amixer", "-q", "sset", "Master", f"{level}%"])
#     return f"громкость {level}%"
