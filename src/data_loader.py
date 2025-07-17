# src/data_loader.py
import json
import logging
import pandas as pd

class DataLoader:
    """Xử lý việc tải tất cả các file dữ liệu cần thiết cho dự án."""
    def __init__(self, config: dict):
        self.paths = config['paths'] 
        logging.info("DataLoader initialized.")
    
    def load_law_corpus(self) -> list[dict]:
        """
        Tải và làm phẳng kho luật từ file JSON.
        Trả về một list các dictionary, mỗi dict chứa law_id, article_id, và text.
        """
        file_path = self.paths['law_corpus']
        logging.info(f'Loading law corpus from {file_path}')
        with open(file_path, 'r', encoding='utf-8') as f: # Sửa lỗi encoding
            raw_data = json.load(f)
        
        articles = []
        for law in raw_data:
            for article in law['articles']:
                articles.append({
                    "law_id": law['id'],
                    "article_id": article['id'],
                    'text': article['text']
                })
        logging.info(f"Successfully loaded {len(articles)} articles.")
        return articles

    def load_questions(self, data_name: str) -> list[dict]:
        """
        Tải dữ liệu câu hỏi (train hoặc test).
        :param data_name: key của file trong config, ví dụ 'train_data' hoặc 'test_data'.
        """
        file_path = self.paths[data_name]
        logging.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f'Successfully loaded {len(data)} questions.')
        return data
    
    def load_stopwords(self) -> set:
        """Tải stopwords từ file."""
        file_path = self.paths.get('stopwords') # Dùng .get() để tránh lỗi nếu không có
        if not file_path:
            return set()

        logging.info(f"Loading stopwords from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f if line.strip()}
            logging.info(f"Successfully loaded {len(stopwords)} stopwords.")
            return stopwords
        except FileNotFoundError:
            logging.warning(f"Could not find stopwords file at {file_path}")
            return set()