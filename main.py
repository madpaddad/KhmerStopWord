from preprocessing.ps01_normalization import Normalizer
from preprocessing.ps02_tokenizer import Tokenizer
from utils.loaders import load_khmer_dictionary


class Pipeline:
    def __init__(self, text: str, dictionary: set):
        self.text = text
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer(dictionary=dictionary)
    
    def run(self) -> list[str]:
        normalized_text = self.normalizer.process(self.text)
        tokens = self.tokenizer.process(normalized_text)
        return tokens

if __name__ == "__main__":
    # Load dictionary
    dictionary = load_khmer_dictionary()
    text = "ខ្ញុំ!!!   ជានិស្សិត@ITC"

    pipeline = Pipeline(text=text, dictionary=dictionary)
    tokens = pipeline.run()
    print(tokens)