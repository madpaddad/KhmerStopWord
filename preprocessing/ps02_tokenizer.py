from config.constants import DEFAULT_MAX_WORD_LEN
from preprocessing.base import PreprocessorBase

class Tokenizer(PreprocessorBase):
    def __init__(self, dictionary: set = None, max_word_length: int = DEFAULT_MAX_WORD_LEN):
        self.dictionary = dictionary or set()
        self.max_word_length = max_word_length

    def process(self, text: str) -> list[str]:
        tokens = []
        for chunk in text.split():  # simple whitespace split
            tokens.extend(self._tokenize_chunk(chunk))
        return tokens

    def _tokenize_chunk(self, chunk: str) -> list[str]:
        tokens = []
        i = 0
        n = len(chunk)
        while i < n:
            matched = False
            for size in range(self.max_word_length, 0, -1):
                if i + size > n:
                    continue
                word = chunk[i:i+size]
                if word in self.dictionary:
                    tokens.append(word)
                    i += size
                    matched = True
                    break

            # Fallback: keep Latin words together
            if not matched:
                if chunk[i].isascii():
                    start = i
                    while i < n and chunk[i].isascii():
                        i += 1
                    tokens.append(chunk[start:i])
                else:
                    tokens.append(chunk[i])
                    i += 1

        return tokens