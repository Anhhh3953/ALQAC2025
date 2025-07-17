# evaluate.py
import os
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
        # Tạo thư mục cha nếu nó chưa tồn tại
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Thêm FileHandler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '%(asctime)s [%(levelname)s] - %(name)s - %(message)s'),
        handlers=handlers
    )

def calculate_all_metrics(predicted_dicts, ground_truth_dicts):
    """Tính Precision, Recall, và F2-score."""
    predicted_set = {(item['law_id'], item['article_id']) for item in predicted_dicts}
    truth_set = {(item['law_id'], item['article_id']) for item in ground_truth_dicts}
    
    if not predicted_set: return 0.0, 0.0, 0.0
    
    true_positives = len(predicted_set.intersection(truth_set))
    
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(truth_set) if len(truth_set) > 0 else 1.0
    
    if (4 * precision + recall) == 0:
        return precision, recall, 0.0
    
    f2 = (5 * precision * recall) / (4 * precision + recall)
    return precision, recall, f2

def get_ranked_predictions(query, config, bm25, semantic_retriever, reranker, corpus_map):
    """
    Hàm lõi để chạy toàn bộ pipeline (Retrieve -> Fuse -> Rerank)
    và trả về một danh sách các kết quả đã được xếp hạng.
    """
    # Bước 1: Truy xuất
    bm25_results = bm25.retrieve(query, top_k=config['pipeline_params']['retrieval_top_k'])
    semantic_results = semantic_retriever.retrieve(query, top_k=config['pipeline_params']['retrieval_top_k'])

    # Chuyển đổi format để fusion
    bm25_ids = [f"{d['law_id']}_{d['article_id']}" for d in bm25_results]
    semantic_ids = [f"{d['law_id']}_{d['article_id']}" for d in semantic_results]
    
    # Bước 2: Kết hợp
    fused_ids = reciprocal_rank_fusion([bm25_ids, semantic_ids])
    
    # Bước 3: Xếp hạng lại (nếu có)
    if reranker:
        candidates_for_rerank = []
        for full_id in fused_ids[:config['pipeline_params']['rerank_top_k']]:
            try:
                law_id, article_id = full_id.rsplit('_', 1)
                candidates_for_rerank.append({
                    'law_id': law_id,
                    'article_id': article_id,
                    'text': corpus_map.get(full_id, "")
                })
            except ValueError:
                continue
        # reranker.rerank trả về list các dict đã được xếp hạng
        return reranker.rerank(query, candidates_for_rerank)
    else: # Trường hợp 'hybrid'
        # Chuyển đổi format về lại dict
        final_ranked_dicts = []
        for full_id in fused_ids:
            try: 
                law_id, article_id = full_id.rsplit('_', 1)
                final_ranked_dicts.append({'law_id': law_id, 'article_id': article_id})
            except ValueError:
                continue
        return final_ranked_dicts

def main(args):
    """Hàm chính để chạy pipeline đánh giá hoặc tối ưu hóa."""
    with open("config/config.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config)
    
    # 1. Load Data và khởi tạo các thành phần
    logging.info("--- Giai đoạn 1: Tải dữ liệu và Khởi tạo các thành phần ---")
    data_loader = DataLoader(config)
    train_full_data = data_loader.load_questions(data_name='train_data')
    stopwords = data_loader.load_stopwords()
    
    _, val_data = train_test_split(train_full_data, test_size=0.2, random_state=42)
    logging.info(f"Sẽ chạy trên {len(val_data)} mẫu từ tập validation.")

    preprocessor = TextPreprocessor(stopwords=stopwords)
    bm25 = BM25Retriever(model_path=config['paths']['bm25_model'], preprocessor=preprocessor)
    bm25.load_model()
    
    semantic_retriever = SemanticRetriever(model_name=config['models']['bi_encoder'], model_path=config['paths']['semantic_model'])
    semantic_retriever.load_model()
    
    reranker = ReRanker(model_path=config['paths']['reranker_model']) if args.method == 'rerank' else None
    
    corpus_map = {f"{doc['law_id']}_{doc['article_id']}": doc['text'] for doc in data_loader.load_law_corpus()}

    # 2. Lấy dự đoán xếp hạng cho toàn bộ tập validation (chạy 1 lần duy nhất)
    logging.info(f"--- Giai đoạn 2: Lấy dự đoán xếp hạng với phương pháp '{args.method.upper()}' ---")
    all_predictions = []
    for item in tqdm(val_data, desc="Getting ranked predictions"):
        ranked_preds = get_ranked_predictions(item['text'], config, bm25, semantic_retriever, reranker, corpus_map)
        all_predictions.append({'true': item['relevant_articles'], 'pred_ranked': ranked_preds})

    # 3. Chạy đánh giá hoặc tối ưu hóa
    if args.optimize_threshold:
        logging.info("--- Giai đoạn 3: Tối ưu hóa ngưỡng (threshold) ---")
        best_f2, best_k = -1, -1
        for k in tqdm(range(1, 51), desc="Optimizing K"):
            f2_scores = [calculate_all_metrics(item['pred_ranked'][:k], item['true'])[2] for item in all_predictions]
            macro_f2 = np.mean(f2_scores)
            if macro_f2 > best_f2:
                best_f2, best_k = macro_f2, k
        
        logging.info("----------- Optimization Finished -----------")
        logging.info(f"Method: {args.method.upper()}")
        logging.info(f"==> Ngưỡng tốt nhất (Best Threshold): {best_k}")
        logging.info(f"==> Điểm Macro F2-Score cao nhất: {best_f2:.4f}")
    else:
        logging.info(f"--- Giai đoạn 3: Đánh giá với ngưỡng cố định K={args.threshold} ---")
        metrics = [calculate_all_metrics(item['pred_ranked'][:args.threshold], item['true']) for item in all_predictions]
        avg_precision = np.mean([m[0] for m in metrics])
        avg_recall = np.mean([m[1] for m in metrics])
        avg_f2 = np.mean([m[2] for m in metrics])
        
        logging.info("----------- Evaluation Finished -----------")
        logging.info(f"Method: {args.method.upper()}")
        logging.info(f"Final Threshold (Top-N): {args.threshold}")
        logging.info(f"Average Precision: {avg_precision:.4f}")
        logging.info(f"Average Recall: {avg_recall:.4f}")
        logging.info(f"Macro F2-Score: {avg_f2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy đánh giá hoặc tối ưu hóa pipeline trên tập validation.")
    parser.add_argument('--method', type=str, choices=['hybrid', 'rerank'], required=True, help="Phương pháp để đánh giá.")
    parser.add_argument('--threshold', type=int, help="Ngưỡng cố định (top-N). Bắt buộc nếu không có --optimize-threshold.")
    parser.add_argument('--optimize-threshold', action='store_true', help="Thêm cờ này để tự động tìm ngưỡng tốt nhất.")
    
    args = parser.parse_args()
    if not args.optimize_threshold and args.threshold is None:
        parser.error("Cần cung cấp --threshold khi không sử dụng --optimize-threshold.")
        
    main(args)