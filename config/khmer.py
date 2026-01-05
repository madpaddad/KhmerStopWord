import re

# Khmer Unicode block
KHMER_RANGE = re.compile(r"[\u1780-\u17FF]+")

# Khmer punctuation
KHMER_PUNCTUATION = "។៕៚៛៙"

# Characters to ignore
IGNORED_CHARS = set("0123456789")