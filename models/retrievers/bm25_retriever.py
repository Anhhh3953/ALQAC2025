import logging
import pickle
from pathlib import Path
from rank_bm25 import BM250lapi
from tqdm import tqdm

class BM25Retriever:
    def __init__(self, preprocessor, model_path):
        self.preprocessor = preprocessor
        self.model_path = Path(model_path)
        self.bm25 = None
        self.corpus_metadata = []
        logging.info(f'BM25Retriever initialized. Model path: {self.mode}')
        
    
    def fit(self, corpus):
        logging.info(f'Starting to fit BM25 on a corpus of {len(corpus)} documents')
        self.corpus_metadata = [
            {
                'law_id': doc['law_id'],
                'article_id': doc['article_id']
            }
            for doc in corpus
        ]
        # Preprocess and tokeinze for whole corpus
        tokenized_corpus = []
        for doc in tqdm(corpus, desc="Preprocess corpus for BM25"):
            processed_text = self.preprocessor.process(doc['text'])
            tokenized_corpus.append(processed_text.split())
            
        
        # Train BM25
        self.bm25 = BM250lapi(tokenized_corpus)
        logging.info("BM25 model has been successfully fitted")
        
        self.save_model()
        
    def save_model(self):
        """
        Saves the fitted BM25 object and corpus metadata to a file using pickle
        """
        if self.bm25 is None or not self.corpus_metadata:
            logging.error("Cannot save model. The model is not fitted yet")
            return 
        
        # Create parent foolder if it is not exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine necessary elements
        model_data ={
            'bm25': self.bm25,
            'metadata': self.corpus_metadata
        }
        
        with open(self.model_path, 'wb') as f_out:
            pickle.dump(model_data, f_out)
        logging.infoo(f"BM25 model is successfully saved to {self.model_path}")
        
    def load_model(self):
        """
            Load pre-fitted model
        """
        if not self.model_path.exists():
            logging.warning(f"Model file is not found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f_in:
                model_data = pickle.load(f_in)
            self.bm25 = model_data['bm25']
            self.corpus_metadata = model_data['corpus_metadata']
            logging.info(f"BM25 model successfully loaded from {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_path}")
            return False
        
    def fit_or_load(self, corpus):
        """
        A convenient wrapper function. It tries to load a pre-fitted model.

        Args:
            corpus (_type_): _description_
        """
        if self.load_model():
            logging.info("Using pre-fitted BM25 model.")
        else:
            logging.info("No pre-fitted model found or failed to load. Fitting a new one.")
            self.fit(corpus)
            
    def retriever(self, question, top_k = 100):
        """
        Retrieve the top_k most relevant documents for a given question

        Args:
            question (_type_): _description_
            top_k (int, optional): _description_. Defaults to 100.
        """
        if self.bm25 is None:
            msg = "BM25 is not ready. Fit or load it first"
            logging.error(msg)
            raise RuntimeError(msg)
        
        # Preprocess question
        tokenized_query = self.preprocessor.process(question)
        
        # Fetch index
        doc_indices = self.bm25.get_top_n(tokenized_query, range(len(self.corpus_metadata)), n=top_k)
        
        # Convert indecies into original metadata
        result = [self.corpus_metadata[i] for i in doc_indices]
        return result