
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLAMA = ROOT /".."/ "llama"

with open(LLAMA / "arch.json", "r", encoding="utf-8") as file:
    config = json.load(file)

architecture = config["architecture"]
llama_server = LLAMA / config["binary"]

print("Архитектура:", architecture)
print("LLAMA Server:", llama_server)

if llama_server.exists():
    print("Сервер найден!")
else:
    print("Сервер не найден!")
