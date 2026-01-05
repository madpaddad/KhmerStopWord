import csv
import unicodedata
from config.paths import KHMER_DICT_CSV
from config.constants import UTF8


def load_khmer_dictionary() -> set[str]:
    dictionary = set()
    with open(KHMER_DICT_CSV, mode='r', encoding=UTF8) as file:
        reader = csv.DictReader(file)
        for row in reader:
            word = unicodedata.normalize("NFC", row["word"].strip())
            dictionary.add(word)

    return dictionary

def load_khmer_pos_dictionary() -> dict[str, str]:
    lexicon = {}
    with open(KHMER_DICT_CSV, mode='r', encoding=UTF8) as file:
        reader = csv.DictReader(file)
        for row in reader:
            word = unicodedata.normalize("NFC", row["word"].strip())
            pos = row["pos"].strip()
            lexicon[word] = pos

    return lexicon
