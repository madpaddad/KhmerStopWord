import re
import unicodedata
from preprocessing.base import PreprocessorBase
from typing import List, Tuple, Optional, Union

class NER(PreprocessorBase):
    def __init__(
        self, 
        person_names: set[str] = None,
        org_names: set[str] = None,
        locations: set[str] = None
    ):
        self.person_names = {unicodedata.normalize("NFC", w) for w in (person_names or set())}
        self.org_names = {unicodedata.normalize("NFC", w) for w in (org_names or set())}
        self.locations = {unicodedata.normalize("NFC", w) for w in (locations or set())}

        # regex patterns
        self.number_pattern = re.compile(r'^[0-9០១២៣៤៥៦៧៨៩]+$')
        self.latin_pattern = re.compile(r'^[A-Za-z]+$')
        self.email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

    def process(self, tokens: Union[List[str], List[Tuple[str, str]]]) -> List[Tuple[str, str, Optional[str]]]:
        """
        tokens: list of strings OR list of (token, POS) tuples
        Returns: list of (token, POS, NER-tag) if POS exists, else (token, None, NER-tag)
        """
        tagged = []

        for item in tokens:
            if isinstance(item, tuple):
                token, pos = item
            else:
                token, pos = item, None

            token_norm = unicodedata.normalize("NFC", token)

            if token_norm in self.person_names:
                ner_tag = "PERSON"
            elif token_norm in self.org_names:
                ner_tag = "ORG"
            elif token_norm in self.locations:
                ner_tag = "LOC"
            elif self.number_pattern.match(token_norm):
                ner_tag = "NUMBER"
            elif self.latin_pattern.match(token_norm):
                ner_tag = "LATIN"
            elif self.email_pattern.match(token_norm):
                ner_tag = "EMAIL"
            else:
                ner_tag = "O"  # Outside any named entity

            tagged.append((token, pos, ner_tag))

        return tagged