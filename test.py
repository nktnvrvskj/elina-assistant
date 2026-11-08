from pathlib import Path
from core.Config import Config
from core.Llama import Llama
from core.Api import Api
from core.WakeWord import WakeWord


class Test:
    def __init__(self):
        self.config = Config()
        self.llama = Llama(self.config)
        self.api = Api(self.config)
        self.wake = None
        self.main()

    def main(self):
        actions = {
            "1": self.set_model, "2": self.llama.start,
            "3": self.llama.stop, "4": self.chat,
            "5": lambda: print(self.llama.read_output()), "6": self.voice,
        }
        while True:
            model = Path(self.config.models_path).name if self.config.models_path else "-"
            print(f"\n[{'RUN' if self.llama.is_running() else 'STOP'}] {model}")
            print("1 model | 2 start | 3 stop | 4 chat | 5 log | 6 voice | 0 exit")
            c = input("> ").strip()
            if c == "0":
                return
            try:
                actions[c]()
            except KeyError:
                print("?")
            except Exception as e:
                print(f"ERR: {e}")

    def set_model(self):
        self.llama.get_models()
        if not self.llama.models:
            print("no .gguf found")
            return
        for i, m in enumerate(self.llama.models):
            print(f"  {i}) {m}")
        raw = input("index: ").strip()
        if raw.isdigit() and int(raw) < len(self.llama.models):
            self.config.post_models(self.llama.models[int(raw)])
            print("ok")

    def chat(self):
        if not self.llama.is_running():
            print("not running")
            return
        print(self.api.send_ai(input("you: ")))

    def voice(self):
        if self.wake is None:
            self.wake = WakeWord()
        print("Скажите «Элина»…")
        text = self.wake.listen_command()
        if not text:
            print("не расслышал")
            return
        print(f"you: {text}")
        if not self.llama.is_running():
            print("not running")
            return
        print(self.api.send_ai(text))


if __name__ == "__main__":
    Test()