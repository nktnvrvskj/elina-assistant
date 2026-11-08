from core.config import Config
from core.llama import Llama
from core.api import Api
from core.agent import Agent
from core.voice import Voice
from core.setup import prepare_files, run as run_setup

EXIT_WORDS = {"пока", "выход", "стоп", "хватит", "до свидания", "завершить"}


class Assistant:
    """Консольное меню: модель, сервер, чат, голос и настройки."""

    def __init__(self, setup_only=False):
        self.config = Config()
        prepare_files(self.config)
        self.llama = Llama(self.config)

        if setup_only or not self.config.setup_done:
            run_setup(self.config, self.llama)
            if setup_only:
                print("Готово. Запустите: python main.py")
                return

        self.api = Api(self.config)
        self.agent = Agent(self.config, self.api)
        self.voice = Voice(self.config)
        if self.config.autostart_llm:
            self.autostart()
        self.menu()

    def autostart(self):
        if self.llama.is_ready():
            return
        print("авто-запуск llama-server…")
        try:
            self.llama.start()
        except Exception as error:
            print(f"не удалось запустить: {error}")

    def menu(self):
        actions = {
            "1": self.choose_model,
            "2": self.llama.start,
            "3": self.stop_server,
            "4": self.chat,
            "5": self.show_log,
            "6": self.voice_mode,
            "7": self.choose_voice,
            "8": self.toggle_autostart,
            "9": self.settings,
        }
        while True:
            state = "работает" if self.llama.is_ready() else "остановлен"
            engine = self.voice.engine or "нет"
            autostart = "вкл" if self.config.autostart_llm else "выкл"
            thinking = "выкл"
            if self.config.thinking:
                budget = self.config.thinking_budget
                thinking = "вкл, без лимита" if budget < 0 else f"вкл, {budget} ток."
            print(f"\n[сервер {state}] модель: {self.config.model or 'не выбрана'}")
            print(f"голос: {engine} | авто-старт LLM: {autostart} | thinking: {thinking}")
            print("1 модель | 2 запуск | 3 стоп | 4 чат | 5 лог | 6 голос")
            print("7 озвучка | 8 авто-старт LLM | 9 настройка | 0 выход")
            try:
                choice = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self.exit()
                return
            if choice == "0":
                self.exit()
                return
            action = actions.get(choice)
            if action is None:
                print("нет такого пункта")
                continue
            try:
                action()
            except Exception as error:
                print(f"ошибка: {error}")

    def exit(self):
        self.llama.stop()
        self.voice.close()

    def choose_model(self):
        models = self.llama.get_models()
        if not models:
            print("в llama/gguf нет файлов .gguf")
            return
        for index, name in enumerate(models):
            print(f"  {index}) {name}")
        raw = input("номер: ").strip()
        if raw.isdigit() and int(raw) < len(models):
            self.config.select_model(models[int(raw)])
            print(f"выбрана модель: {self.config.model}")
        else:
            print("неверный номер")

    def show_log(self):
        line = self.llama.read_output()
        if not line:
            print("(новых строк нет)")
            return
        while line:
            print(line)
            line = self.llama.read_output()

    def stop_server(self):
        if not self.llama.is_running() and self.llama.is_ready():
            print("сервер запущен не из этого меню — остановите его вручную")
            return
        self.llama.stop()

    def chat(self):
        if not self.llama.is_ready():
            print("сервер не запущен (пункт 2)")
            return
        print(self.agent.ask(input("вы: ")))

    def choose_voice(self):
        print("озвучка: 0 авто | 1 piper | 2 espeak-ng | 3 выключить")
        print(f"сейчас: {self.config.voice_engine} → {self.voice.engine or 'недоступно'}")
        choice = input("> ").strip()
        engines = {"0": "auto", "1": "piper", "2": "espeak-ng", "3": "none"}
        if choice not in engines:
            print("нет такого пункта")
            return
        self.config.set_voice(engines[choice])
        self.voice.close()
        self.voice = Voice(self.config)
        if self.voice.engine:
            print(f"озвучка: {self.voice.engine}")
        else:
            print(f"озвучка недоступна: {self.voice.problem}")

    def toggle_autostart(self):
        self.config.set_autostart(not self.config.autostart_llm)
        state = "включён" if self.config.autostart_llm else "выключен"
        print(f"авто-запуск LLM: {state}")

    def settings(self):
        run_setup(self.config, self.llama)
        self.voice.close()
        self.voice = Voice(self.config)
        if self.config.autostart_llm and not self.llama.is_ready():
            self.autostart()

    def voice_mode(self):
        if not self.llama.is_ready():
            print("сервер не запущен (пункт 2)")
            return
        print("Скажите «Элина»… Ctrl+C — выход, «пока» — закончить")
        try:
            while True:
                text = self.voice.listen_command()
                if not text:
                    continue
                print(f"вы: {text}")
                if self._is_exit(text):
                    self.voice.say("До встречи!")
                    return
                answer = self.agent.ask(text)
                print(answer)
                self.voice.say(answer)
        except KeyboardInterrupt:
            print("\nвыход из голосового режима")

    @staticmethod
    def _is_exit(text):
        words = text.lower().replace("ё", "е").strip(" .,!?")
        return words in EXIT_WORDS


if __name__ == "__main__":
    import sys

    Assistant(setup_only="--setup" in sys.argv)
