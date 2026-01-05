

from abc import ABC, abstractmethod


class PreprocessorBase(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")