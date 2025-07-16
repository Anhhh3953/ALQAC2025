import re
import logging
from pyvi import ViTokenizer

class TextPreprocessor:
    def __init__(self, stopwords):
        self.stopwords = stopwords
        self.punctuation_re = re.compile(r'[^\w\s]')
        logging.infoo('TextPreprocessor initialized')

    def _load_stopwords(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f if line.strip()}
            logging.info(f"Successfully loaded {len(stopwords)} stopwords.")
            return stopwords
        except FileNotFoundError:
            logging.warning(f"Stopwords file not found at {path}. Proceeding with an empty set.")
            return set()   
    
    def process(self, text):
        """
        Applies the full preprocessing pipeline to a given text.

        The steps are:
        1. Convert to lowercase.
        2. Remove punctuation.
        3. Tokenize using PyVi for Vietnamese word segmentation.
        4. Remove stopwords and short tokens.
        """
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove newlines and extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 3. Remove punctuation
        text = self.punctuation_re.sub(' ', text)
        
        # 4. Tokenize with pyvi
        tokenized_text = ViTokenizer.tokenize(text)
        
        # 5. Eliminate stopwords
        tokens = tokenized_text.split()
        tokens = [token for token in tokenized_text if token not in self.stopwords and len(token) > 1]
        return tokens
    
        