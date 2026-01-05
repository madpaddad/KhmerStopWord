from typing import List, Set, Dict, Tuple
import unicodedata

from preprocessing.ps01_normalization import Normalizer
from preprocessing.ps02_tokenizer import Tokenizer
from preprocessing.ps03_pos_tagger import POSTagger
from preprocessing.ps04_stop_word import StopWordRemover
from preprocessing.ps05_ner import NER
from preprocessing.ps06_frequency import FrequencyAnalyzer
from preprocessing.ps07_contextual_scorer import ContextualScorer

class KhmerStopWordPipeline:
    def __init__(
        self,
        dictionary: Set[str],
        pos_lexicon: Dict[str, str] = None,
        stop_words: Set[str] = None,
        stop_pos: Set[str] = None,
        person_names: Set[str] = None,
        org_names: Set[str] = None,
        locations: Set[str] = None
    ):
        # Core preprocessing
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer(dictionary=dictionary)
        self.pos_tagger = POSTagger(lexicon=pos_lexicon)
        self.stop_word_remover = StopWordRemover(stop_words=stop_words, stop_pos=stop_pos)
        self.ner = NER(person_names=person_names, org_names=org_names, locations=locations)
        
        # Frequency / Contextual scorer
        self.freq_analyzer = FrequencyAnalyzer(dictionary=dictionary)
        self.contextual_scorer = ContextualScorer()
    
    def run_pipeline(self, text: str) -> List[Tuple[str, str, str]]:
        # Step 1: normalize
        normalized = self.normalizer.process(text)
        
        # Step 2: tokenize
        tokens = self.tokenizer.process(normalized)
        tokens = [unicodedata.normalize("NFC", t) for t in tokens]
        
        # Step 3: POS tagging
        pos_tagged = self.pos_tagger.process(tokens)
        
        # Step 4: remove stop-words
        cleaned_tokens = self.stop_word_remover.process(pos_tagged)
        
        # Step 5: NER
        ner_tagged = self.ner.process(cleaned_tokens)
        
        # Step 6: Update frequency
        self.freq_analyzer.counter.update([t[0] for t in ner_tagged])
        
        return ner_tagged
    
    def fit_contextual_stopwords(self, corpus: List[str], percentile: float = 20):
        """
        Fit TF*IDF-based contextual stop-words from a corpus.
        """
        all_docs_tokens = []
        for text in corpus:
            normalized = self.normalizer.process(text)
            tokens = self.tokenizer.process(normalized)
            tokens = [unicodedata.normalize("NFC", t) for t in tokens]
            all_docs_tokens.append(tokens)
        
        # Fit TF*IDF
        self.contextual_scorer.fit(all_docs_tokens)
        tfidf_scores = self.contextual_scorer.score_all(all_docs_tokens)
        
        # Suggest stop-words dynamically
        suggested_stopwords = self.contextual_scorer.suggest_stop_words(tfidf_scores, percentile=percentile)
        self.stop_word_remover.stop_words.update(suggested_stopwords)
        
        return suggested_stopwords