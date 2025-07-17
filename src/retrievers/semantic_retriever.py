import logging
import pickle
import faiss
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class SemanticRetriever:
    def __init__(self, model_name: str, model_path: str):
        """
        Khởi tạo Semantic Retriever.
        :param model_name: Tên của mô hình Bi-Encoder từ Hugging Face (ví dụ: 'bkai-foundation-models/vietnamese-bi-encoder').
        :param model_path: Đường dẫn để lưu/tải các artifacts (chỉ mục FAISS và metadata).
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.model_path = Path(model_path)
        self.index = None
        self.corpus_metadata = []
        logging.info(f'SemanticRetriever initialized with model "{model_name}" on device "{self.device}".')

    def fit(self, corpus: list[dict]):
        """
        Vector hóa toàn bộ kho luật và xây dựng chỉ mục FAISS.
        :param corpus: Một list các dictionary, mỗi dict chứa 'law_id', 'article_id', và 'full_text'.
        """
        logging.info(f'Starting to fit Semantic Retriever on a corpus of {len(corpus)} documents.')

        # Lưu lại metadata để có thể trả về đúng law_id và article_id
        self.corpus_metadata = [{'law_id': doc['law_id'], 'article_id': doc['article_id']} for doc in corpus]
        
        # Lấy ra toàn bộ văn bản để vector hóa
        full_texts = [doc['full_text'] for doc in corpus]
        
        logging.info("Encoding corpus... This may take a while.")
        embeddings = self.encoder.encode(
            full_texts,
            show_progress_bar=True,
            batch_size=128,  # Có thể điều chỉnh batch size tùy thuộc vào VRAM của GPU
            convert_to_numpy=True
        )
        
        logging.info("Building FAISS index...")
        d = embeddings.shape[1]  # Lấy chiều của vector
        self.index = faiss.IndexFlatIP(d)
        
        # Chuẩn hóa vector trước khi thêm vào chỉ mục IndexFlatIP
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        logging.info(f"FAISS index successfully built with {self.index.ntotal} vectors.")
        
        self.save_model()

    def save_model(self):
        """Lưu chỉ mục FAISS và metadata."""
        if self.index is None or not self.corpus_metadata:
            logging.error("Cannot save model. The model is not fitted yet.")
            return
            
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu chỉ mục FAISS
        faiss_index_path = self.model_path.with_suffix('.index')
        faiss.write_index(self.index, str(faiss_index_path))
        
        # Lưu metadata
        metadata_path = self.model_path.with_suffix('.meta.pkl')
        with open(metadata_path, 'wb') as f_out:
            pickle.dump(self.corpus_metadata, f_out)
            
        logging.info(f"Semantic model artifacts successfully saved to {self.model_path.parent}")

    def load_model(self):
        """Load chỉ mục FAISS và metadata đã được tạo trước."""
        faiss_index_path = self.model_path.with_suffix('.index')
        metadata_path = self.model_path.with_suffix('.meta.pkl')

        if not faiss_index_path.exists() or not metadata_path.exists():
            logging.warning(f"Model artifacts not found at {self.model_path.parent}. Please fit the model first.")
            return False
            
        try:
            logging.info(f"Loading FAISS index from {faiss_index_path}...")
            self.index = faiss.read_index(str(faiss_index_path))
            
            logging.info(f"Loading metadata from {metadata_path}...")
            with open(metadata_path, 'rb') as f_in:
                self.corpus_metadata = pickle.load(f_in)
                
            logging.info(f"Semantic model successfully loaded.")
            return True
        except Exception as e:
            logging.error(f"Failed to load semantic model artifacts: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 100) -> list[dict]:
        """
        Truy xuất top_k tài liệu liên quan nhất.
        :return: Một list các dictionary, mỗi dict chứa 'law_id' và 'article_id'.
        """
        if self.index is None:
            msg = "Semantic model is not ready. Please load a model first."
            logging.error(msg)
            raise RuntimeError(msg)
            
        # Vector hóa câu hỏi
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        k = min(top_k, self.index.ntotal)
        
        # Tìm kiếm trong chỉ mục FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Chuyển các chỉ số thành metadata
        results = [self.corpus_metadata[i] for i in indices[0]]
        return results