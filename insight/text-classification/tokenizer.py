class CharacterTokenizer:
    def __init__(self):
        self.tokenizer = {
            "<pad>": 0
        }
        self.count = 1

    def __call__(self, x):
        return self.encode(x)

    def train_char(self, c):
        if c not in self.tokenizer:
            self.tokenizer[c] = self.count
            self.count += 1

    def train(self, s):
        for c in s:
            self.train_char(c)

    def encode_char(self, c):
        return self.tokenizer[c]

    def encode(self, s):
        return [self.encode_char(c) for c in s]

    def decode_token(self, token):
        return self.tokenizer.keys()[token]

    def decode(self, tokens):
        return [self.decode_token(token) for token in tokens]
