from collections import Counter
from preprocessing.ps01_normalization import Normalizer
from preprocessing.ps02_tokenizer import Tokenizer
import unicodedata
from typing import List, Set, Dict

class FrequencyAnalyzer:
    def __init__(self, dictionary: Set[str]):
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer(dictionary=dictionary)
        self.counter = Counter()

    def process_text(self, text: str):
        normalized = self.normalizer.process(text)
        tokens = self.tokenizer.process(normalized)
        tokens = [unicodedata.normalize("NFC", t) for t in tokens]
        self.counter.update(tokens)

    def process_corpus(self, texts: List[str]):
        for text in texts:
            self.process_text(text)

    def most_common(self, n: int = 20) -> Dict[str, int]:
        return dict(self.counter.most_common(n))

    def get_frequency(self, token: str) -> int:
        token_norm = unicodedata.normalize("NFC", token)
        return self.counter[token_norm]

    def save_to_file(self, filepath: str):
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.counter, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str):
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            self.counter = Counter(json.load(f))