
import unicodedata
from preprocessing.base import PreprocessorBase
from utils.loaders import load_khmer_pos_dictionary


class POSTagger(PreprocessorBase):
    
    def __init__(self, lexicon: dict = None):
        """
        dictionary: dict[word] = POS
        """
        self.lexicon = lexicon or load_khmer_pos_dictionary()
        self.lexicon = { unicodedata.normalize("NFC", word) : pos for word, pos in self.lexicon.items()}
    
    def process(self, tokens: list[str]) -> list[tuple[str, str]]:
        tagged = []
        for token in tokens:
            token_norm = unicodedata.normalize("NFC", token)
            pos = self.lexicon.get(token_norm, "UNK")
            tagged.append((token, pos))
        return tagged
