# src/rankers.py
import torch
import logging
from sentence_transformers import CrossEncoder

class ReRanker:
    """
    Sử dụng một mô hình Cross-Encoder đã được fine-tune để xếp hạng lại
    các ứng viên từ giai đoạn truy xuất.
    """
    def __init__(self, model_path: str, max_length: int = 256):
        """
        Khởi tạo ReRanker.
        :param model_path: Đường dẫn đến thư mục chứa model đã fine-tune.
        :param max_length: Độ dài tối đa của chuỗi đầu vào.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_path, max_length=max_length, device=self.device)
        logging.info(f"ReRanker model loaded from '{model_path}' on device '{self.device}'.")

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """
        Xếp hạng lại một danh sách các tài liệu.
        
        :param query: Câu hỏi.
        :param documents: List các dictionary, mỗi dict chứa 'law_id', 'article_id', 'text'.
        :return: List các dictionary đã được sắp xếp lại, mỗi dict chứa 'law_id' và 'article_id'.
        """
        if not documents:
            return []
            
        # Tạo các cặp (câu hỏi, văn bản) để đưa vào model
        pairs = [[query, doc.get('text', '')] for doc in documents]
        
        # Dự đoán điểm số
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Gắn điểm số vào các tài liệu ban đầu
        for doc, score in zip(documents, scores):
            doc['score'] = score
        
        # Sắp xếp các tài liệu dựa trên điểm số giảm dần
        sorted_documents = sorted(documents, key=lambda x: x.get('score', 0.0), reverse=True)
        
        # Trả về kết quả với định dạng mong muốn, loại bỏ các trường không cần thiết
        final_results = [
            {'law_id': doc['law_id'], 'article_id': doc['article_id']}
            for doc in sorted_documents
        ]
        
        return final_results