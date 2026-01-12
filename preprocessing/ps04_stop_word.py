import unicodedata
from preprocessing.base import PreprocessorBase
from typing import Optional, Set, Union, List, Tuple

class StopWordRemover(PreprocessorBase):
    def __init__(self, stop_words: Optional[Set[str]] = None, stop_pos: Optional[Set[str]] = None):
        self.stop_words = {unicodedata.normalize("NFC", w) for w in (stop_words or set())}
        self.stop_pos = stop_pos or set()

    def process(self, tokens: Union[List[str], List[Tuple[str, str]]]) -> List[str]:
        """
        Remove stop-words from a list of tokens.
        Can handle:
            - List of words: ["ខ្ញុំ", "ជា"]
            - List of (token, POS) tuples: [("ខ្ញុំ", "បុ."), ("ជា", "ន.")]
        """
        cleaned = []
        
        if not tokens:
            return []

        # Detect if POS info is present
        if isinstance(tokens[0], tuple) and len(tokens[0]) == 2:
            for token, pos in tokens:
                token_norm = unicodedata.normalize("NFC", token)
                if token_norm not in self.stop_words and pos not in self.stop_pos:
                    cleaned.append((token, pos))
        else:
            for token in tokens:
                token_norm = unicodedata.normalize("NFC", token)
                if token_norm not in self.stop_words:
                    cleaned.append(token)

        return cleaned