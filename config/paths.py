from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "datasets"

DICT_DIR = DATA_DIR / "dicts"

KHMER_DICT_CSV = DICT_DIR / "khmer_pos.csv"