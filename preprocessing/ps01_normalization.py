"""
Khmer text normalization module.
This module provides functionality to normalize Khmer text by removing special characters
and standardizing whitespace.
"""

import re
import unicodedata

from preprocessing.base import PreprocessorBase


class Normalizer(PreprocessorBase):
    def process(self, text: str) -> str:
        text = re.sub(r"[!@#$%^&*()_+=\[\]{};:'\",.<>/?]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
