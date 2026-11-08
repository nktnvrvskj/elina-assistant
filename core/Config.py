from pathlib import Path
import os
import json

class Config:
    def __init__(self):
        self.path = Path(__file__).resolve().parent.parent
        self.llama_path = self.path / "llama"
        self.conf_file =self.path/"config.json"
        self.host_llama ="127.0.0.1"
        self.port_llama ="8080"
        self.server_host_llama="127.0.0.1"
        self.server_port_llama="8080"
        self.models_path = None
        self.SYSTEM_PROMPT = """
            Ты краткий разговорный помощник.
            Отвечай сразу по существу.
            Не показывай ход рассуждений.
            Не пиши длинных вступлений.
            Обычно отвечай 1-3 предложениями.
        """

    def load(self):
        if not os.path.exists(self.conf_file):
            return

        with open(self.conf_file,"r",encoding="utf-8") as f:
            return
    def save(self):
        with open(self.conf_file, "w", encoding="utf-8") as f:

            return
    def post_models(self,name_model):
        self.models_path = self.llama_path / "gguf" /name_model
