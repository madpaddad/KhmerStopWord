

from abc import abstractmethod


class PreprocessorBase:
    @abstractmethod
    def process(self, text: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")