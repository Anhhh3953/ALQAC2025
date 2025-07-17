# build_artifacts.py
import yaml
import logging
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.semantic_retriever import SemanticRetriever

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    with open("config/config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("--- Bắt đầu xây dựng các Artifacts ---")

    # 1. Sử dụng DataLoader để tải dữ liệu
    data_loader = DataLoader(config)
    corpus = data_loader.load_law_corpus()
    stopwords = data_loader.load_stopwords()

    # 2. Khởi tạo Preprocessor
    preprocessor = TextPreprocessor(stopwords=stopwords)

    # 3. Huấn luyện và lưu BM25 Retriever
    bm25_retriever = BM25Retriever(
        model_path=config['paths']['bm25_model'], 
        preprocessor=preprocessor
    )
    bm25_retriever.fit(corpus)
    
    # 4. Huấn luyện và lưu Semantic Retriever
    logging.info("\n--- Fitting Semantic Retriever ---")
    # Tạo 'full_text' cho semantic retriever
    for doc in corpus:
        doc['full_text'] = f"Văn bản pháp luật: {doc['law_id']}. Điều {doc['article_id']}: {doc['text']}"
        
    semantic_retriever = SemanticRetriever(
        model_name=config['models']['bi_encoder'],
        model_path=config['paths']['semantic_model'] 
    )
    semantic_retriever.fit(corpus)

    logging.info("--- Hoàn tất xây dựng Artifacts! ---")

if __name__ == '__main__':
    main()