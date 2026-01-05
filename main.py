from pipeline import KhmerStopWordPipeline
from preprocessing.ps01_normalization import Normalizer
from preprocessing.ps02_tokenizer import Tokenizer
from preprocessing.ps03_pos_tagger import POSTagger
from preprocessing.ps04_stop_word import StopWordRemover
from preprocessing.ps05_ner import NER
from preprocessing.ps07_contextual_scorer import ContextualScorer
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


# frequency_analyzer.py
from preprocessing.ps06_frequency import FrequencyAnalyzer
dictionary = load_khmer_dictionary()
texts = [
    "ខ្ញុំ ជា និស្សិត",
    "ខ្ញុំ បាន ទៅ សាលា និង មិត្តភក្តិ",
    "ITC ជា មហាវិទ្យាល័យ"
]
freq_analyzer = FrequencyAnalyzer(dictionary=dictionary)
freq_analyzer.process_corpus(texts)
print(freq_analyzer.counter)

# contextual example
documents = [
    ["ខ្ញុំ", "ជា", "និស្សិត"],
    ["ខ្ញុំ", "បាន", "ទៅ", "សាលា", "និង", "មិត្តភក្តិ"],
    ["ITC", "ជា", "មហាវិទ្យាល័យ"]
]

scorer = ContextualScorer()
scorer.fit(documents)  # Fit TF-IDF on corpus

# Compute TF*IDF scores
scores = scorer.score_all(documents)
print("TF*IDF scores:", scores)

# Suggest stop-words (low IDF)
stop_words = scorer.suggest_stop_words(scores, percentile=20)
print("Candidate stop-words:", stop_words)


# running the pipeline
# Example corpus and dictionary
print("=== Running KhmerStopWordPipeline ===")
dictionary = {"ខ្ញុំ", "ជា", "និស្សិត", "បាន", "ទៅ", "សាលា", "និង", "មិត្តភក្តិ", "ITC", "មហាវិទ្យាល័យ"}
pos_lexicon = {"ខ្ញុំ": "បុ.", "ជា": "ន.", "និស្សិត": "ន."}

pipeline = KhmerStopWordPipeline(dictionary=dictionary, pos_lexicon=pos_lexicon)

# Fit contextual stop-words from corpus
corpus = [
    "ខ្ញុំ ជា និស្សិត",
    "ខ្ញុំ បាន ទៅ សាលា និង មិត្តភក្តិ",
    "ITC ជា មហាវិទ្យាល័យ"
]
suggested_stopwords = pipeline.fit_contextual_stopwords(corpus, percentile=20)
print("Suggested stop-words:", suggested_stopwords)

# Process single text
text = "ខ្ញុំ!!!   ជានិស្សិត@ITC"
result = pipeline.run_pipeline(text)
print(result)