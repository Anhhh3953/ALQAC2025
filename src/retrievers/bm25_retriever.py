# src/retrievers/bm25_retriever.py
import logging
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi  # Sửa lỗi chính tả
from tqdm import tqdm

class BM25Retriever:
    def __init__(self, model_path: str, preprocessor):
        self.preprocessor = preprocessor
        self.model_path = Path(model_path)
        self.bm25 = None
        self.corpus_metadata = []
        logging.info(f'BM25Retriever initialized. Model path: {self.model_path}')
        
    def fit(self, corpus: list[dict]):
        logging.info(f'Starting to fit BM25 on a corpus of {len(corpus)} documents.')
        
        # Lưu lại metadata để có thể trả về đúng law_id và article_id
        self.corpus_metadata = [{'law_id': doc['law_id'], 'article_id': doc['article_id']} for doc in corpus]
        
        # Tiền xử lý và tách từ cho toàn bộ kho luật
        tokenized_corpus = []
        for doc in tqdm(corpus, desc="Preprocessing corpus for BM25"):
            processed_text = self.preprocessor.process(doc['text'])
            tokenized_corpus.append(processed_text)
            
        # Huấn luyện BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        logging.info("BM25 model has been successfully fitted.")
        
        self.save_model()
        
    def save_model(self):
        """Lưu model BM25 và metadata."""
        if self.bm25 is None or not self.corpus_metadata:
            logging.error("Cannot save model. The model is not fitted yet.")
            return 
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {'bm25': self.bm25, 'metadata': self.corpus_metadata}
        
        with open(self.model_path, 'wb') as f_out:
            pickle.dump(model_data, f_out)
        logging.info(f"BM25 model is successfully saved to {self.model_path}") # Sửa lỗi chính tả
        
    def load_model(self):
        """Load model đã được huấn luyện trước."""
        if not self.model_path.exists():
            logging.warning(f"Model file not found at {self.model_path}. Please fit the model first.")
            return False
        
        try:
            with open(self.model_path, 'rb') as f_in:
                model_data = pickle.load(f_in)
            self.bm25 = model_data['bm25']
            self.corpus_metadata = model_data['metadata']
            logging.info(f"BM25 model successfully loaded from {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_path}: {e}")
            return False
        
    def retrieve(self, query: str, top_k: int = 100) -> list[dict]:
        """Truy xuất top_k tài liệu liên quan nhất."""
        if self.bm25 is None:
            msg = "BM25 model is not ready. Please load a model first."
            logging.error(msg)
            raise RuntimeError(msg)
        
        tokenized_query = self.preprocessor.preprocess(query)
        
        # get_top_n trả về list các document, không phải index.
        # Chúng ta cần dùng get_scores để lấy index hoặc một cách tiếp cận khác.
        # Cách hiệu quả hơn là dùng bm25.get_scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        k = min(top_k, len(doc_scores))
        top_n_indices = np.argsort(doc_scores)[-k:][::-1]
        
        results = [self.corpus_metadata[i] for i in top_n_indices]
        return results
    
    # logic retrieve để dùng get_scores và argsort vì nó hiệu quả và trả về đúng thứ hạng hơn so với get_top_n