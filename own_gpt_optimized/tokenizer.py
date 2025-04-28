import tiktoken

class TikTokenTokenizer:
    def __init__(self, model_name="gpt2"):
        self.enc = tiktoken.get_encoding(model_name)

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, tokens):
        return self.enc.decode(tokens)