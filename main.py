from core.Llama import Llama
from core.Config import Config

class main:
    def __init__(self):
        self.Config = Config()
        self.LLAMA = Llama(self.Config)

        self.LLAMA.get_models()
        print(self.LLAMA.models)

if __name__ == "__main__":
    main()