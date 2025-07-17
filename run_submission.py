# run_submission.py

import yaml
import logging
import argparse
import os
import json
from tqdm import tqdm

# Import các thành phần đã được module hóa
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.semantic_retriever import SemanticRetriever
from src.rankers import ReRanker
from src.fusers import reciprocal_rank_fusion

def setup_logging(config):
    """Cấu hình logging dựa trên file config."""
    log_config = config.get('logging', {})
    log_file = config.get('paths', {}).get('log_file')
    
    handlers = [logging.StreamHandler()] # Luôn in ra console
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '%(asctime)s [%(levelname)s] - %(name)s - %(message)s'),
        handlers=handlers
    )


def main(args):
    """Hàm chính để chạy pipeline và tạo file submission."""
    with open("config/config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config)

    # Lấy thông tin của profile được chọn từ dòng lệnh
    profile = config['submission_profiles'].get(args.profile_name)
    if not profile:
        logging.error(f"Lỗi: Không tìm thấy submission profile '{args.profile_name}' trong config.yaml")
        return

    logging.info(f"--- Bắt đầu tạo submission cho profile: {args.profile_name.upper()} ---")
    
    # 1. Load Data và khởi tạo các thành phần
    logging.info("[1/3] Tải dữ liệu và khởi tạo các thành phần...")
    data_loader = DataLoader(config)
    test_data = data_loader.load_questions(data_name='test_data')
    stopwords = data_loader.load_stopwords()
    
    preprocessor = TextPreprocessor(stopwords=stopwords)
    
    bm25 = BM25Retriever(model_path=config['paths']['bm25_model'], preprocessor=preprocessor)
    if not bm25.load_model(): return
    
    semantic_retriever = SemanticRetriever(model_name=config['models']['bi_encoder'], model_path=config['paths']['semantic_model'])
    if not semantic_retriever.load_model(): return

    reranker = None
    if profile['method'] == 'rerank':
        reranker = ReRanker(model_path=config['paths']['reranker_model'])
    
    corpus_map = {f"{doc['law_id']}_{doc['article_id']}": doc['text'] for doc in data_loader.load_law_corpus()} if reranker else {}

    # 2. Xử lý dữ liệu test
    logging.info(f"[2/3] Xử lý {len(test_data)} câu hỏi từ tập test...")
    submission_results = []
    
    for item in tqdm(test_data, desc=f"Generating submission for '{args.profile_name}'"):
        question_id = item['question_id']
        question_text = item['text']

        # Bước 1: Truy xuất
        bm25_candidates = bm25.retrieve(question_text, top_k=config['pipeline_params']['retrieval_top_k'])
        semantic_candidates = semantic_retriever.retrieve(question_text, top_k=config['pipeline_params']['retrieval_top_k'])
        
        bm25_ids = [f"{d['law_id']}_{d['article_id']}" for d in bm25_candidates]
        semantic_ids = [f"{d['law_id']}_{d['article_id']}" for d in semantic_candidates]

        # Bước 2: Kết hợp
        fused_ids = reciprocal_rank_fusion([bm25_ids, semantic_ids])

        # Bước 3: Xếp hạng lại (nếu có) và áp dụng ngưỡng
        if reranker:
            candidates_for_rerank = []
            for full_id in fused_ids[:config['pipeline_params']['rerank_top_k']]:
                try:
                    law_id, article_id = full_id.rsplit('_', 1)
                    candidates_for_rerank.append({'law_id': law_id, 'article_id': article_id, 'text': corpus_map.get(full_id, "")})
                except ValueError: continue
            
            final_predictions = reranker.rerank(question_text, candidates_for_rerank)
        else: # Trường hợp 'hybrid'
            final_ids = fused_ids
            final_predictions = []
            for full_id in final_ids:
                try: 
                    law_id, article_id = full_id.rsplit('_', 1)
                    final_predictions.append({'law_id': law_id, 'article_id': article_id})
                except ValueError: continue
                
        # Áp dụng ngưỡng cuối cùng từ profile
        final_predictions = final_predictions[:profile['threshold']]
        
        # Bước 4: Định dạng submission
        submission_item = {
            "question_id": question_id,
            "relevant_articles": final_predictions
        }
        submission_results.append(submission_item)
        
    # 3. Ghi file submission
    output_file = profile['output_file']
    logging.info(f"[3/3] Đang ghi file submission vào: {output_file}")
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(submission_results, f, ensure_ascii=False, indent=4)
        
    logging.info(f"✅ Submission file '{output_file}' đã được tạo thành công!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo file submission cho cuộc thi ALQAC 2025.")
    parser.add_argument(
        'profile_name', 
        type=str, 
        help="Tên của submission profile được định nghĩa trong config.yaml (ví dụ: rerank_optimal)."
    )
    args = parser.parse_args()
    main(args)