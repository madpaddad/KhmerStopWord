from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
from typing import List, Dict, Set
import numpy as np


import logging

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

class ContextualScorer:
    """
    Compute contextual stop-word scores for a corpus using TF-IDF.
    Provides methods to get TF*IDF scores and suggest stop-words dynamically.
    """
    def __init__(self):
        self.vectorizer: TfidfVectorizer = None
        self.idf_scores: Dict[str, float] = {}
        self.feature_names: List[str] = []

    def normalize_doc(self, tokens: List[str]) -> str:
        """Normalize a token list into a whitespace-joined string"""
        return " ".join([unicodedata.normalize("NFC", t) for t in tokens])

    def fit(self, documents: List[List[str]]):
        """
        Fit the TF-IDF model on a list of tokenized documents
        """
        # Convert each document into normalized string
        normalized_docs = [self.normalize_doc(doc) for doc in documents]
        

        print('\n\n Fiting the docs with docs')
        for doc in documents:
            print(doc)

        print('\n After normalizing: ', normalized_docs)
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\S+',  # whitespace-safe tokenizer
            use_idf=True,
            smooth_idf=True,
            lowercase=False
        )

        # Fit the vectorizer
        self.vectorizer.fit(normalized_docs)

        # Store IDF scores
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.idf_scores = dict(zip(self.feature_names, self.vectorizer.idf_))

    def score_all(self, documents: List[List[str]]) -> Dict[str, float]:
        """
        Compute TF*IDF scores per token across the corpus
        """
        tf_counter: Dict[str, int] = {}

        for doc in documents:
            for token in doc:
                token_norm = unicodedata.normalize("NFC", token)
                tf_counter[token_norm] = tf_counter.get(token_norm, 0) + 1

        # Compute TF*IDF
        scores = {}
        for token, tf in tf_counter.items():
            idf = self.idf_scores.get(token, max(self.idf_scores.values(), default=1.0))
            scores[token] = tf * idf

        return scores

    def suggest_stop_words(self, tfidf_scores: Dict[str, float], percentile: float = 20) -> Set[str]:
        """
        Suggest stop-words based on the lowest TF*IDF scores
        percentile: select words below the Xth percentile
        """
        if not tfidf_scores:
            raise ValueError("No TF*IDF scores provided for stop-word suggestion.")

        scores_array = np.array(list(tfidf_scores.values()))
        threshold = np.percentile(scores_array, percentile)  # lowest X% are stop-words

        stop_words = {token for token, score in tfidf_scores.items() if score <= threshold}
        return stop_words