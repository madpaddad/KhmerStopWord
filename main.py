from preprocessing.ps01_normalization import Normalizer
from preprocessing.ps02_tokenizer import Tokenizer
from preprocessing.ps03_pos_tagger import POSTagger
from preprocessing.ps04_stop_word import StopWordRemover
from preprocessing.ps05_ner import NER
from utils.loaders import load_khmer_dictionary


class Pipeline:
    def __init__(self, text: str, dictionary: set):
        self.text = text
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer(dictionary=dictionary)
        self.pos_tagger = POSTagger()
        self.stop_word = StopWordRemover(stop_words={"ខ្ញុំ", "ជា"})
        self.ner = NER(locations={"ITC"})
    
    def run(self) -> list[str]:
        normalized_text = self.normalizer.process(self.text)
        tokens = self.tokenizer.process(normalized_text)
        tokens = self.pos_tagger.process(tokens)
        tokens = self.stop_word.process(tokens)
        tokens = self.ner.process(tokens)
        return tokens

if __name__ == "__main__":
    # Load dictionary
    dictionary = load_khmer_dictionary()
    text = "ខ្ញុំ!!!   ជានិស្សិត@ITC"

    pipeline = Pipeline(text=text, dictionary=dictionary)
    tokens = pipeline.run()
    print(tokens)