# run_submission_task2.py

import yaml
import logging
import argparse
import os
import json
import re
from tqdm import tqdm

# Import các thành phần đã được module hóa
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.semantic_retriever import SemanticRetriever
from src.rankers import ReRanker
from src.fusers import reciprocal_rank_fusion
from src.answer_generator import initialize_llm, create_qa_chains, post_process_answer

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

    logging.info(f"--- Bắt đầu tạo submission cho profile: {args.profile_name.upper()} (Task 2: Question Answering) ---")
    
    # 1. Load Data và khởi tạo các thành phần
    logging.info("[1/4] Tải dữ liệu và khởi tạo các thành phần...")
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
    
    # For Task 2:
    logging.info("[2/4 Initialize LLM and QA Chains via LangChain]")
    llm = initialize_llm(model_path=config['paths']['llm_model'], llm_config=config.get('llm_params', {}))
    if not llm:
        logging.error('Failed to initialize LLM')
        return 
    qa_chains = create_qa_chains(llm)
    
    # 2. Xử lý dữ liệu test
    logging.info(f"[3/4] Xử lý {len(test_data)} câu hỏi từ tập test...")
    submission_results = []
    
    for item in tqdm(test_data, desc=f"Generating submission for '{args.profile_name}'"):
        question_id = item['question_id']
        question_text = item['text']
        question_type = item['question_type', 'Tự luận'] # Tự luận as default

        # Bước 1: Truy xuất
        bm25_candidates = bm25.retrieve(question_text, top_k=config['pipeline_params']['retrieval_top_k'])
        semantic_candidates = semantic_retriever.retrieve(question_text, top_k=config['pipeline_params']['retrieval_top_k'])
        
        bm25_ids = [f"{d['law_id']}_{d['article_id']}" for d in bm25_candidates]
        semantic_ids = [f"{d['law_id']}_{d['article_id']}" for d in semantic_candidates]

        # Bước 2: Kết hợp
        fused_ids = reciprocal_rank_fusion([bm25_ids, semantic_ids])

        # Bước 3: Xếp hạng lại (nếu có) và áp dụng ngưỡng
        if reranker:
            candidates_for_rerank = [{'full_id': fid, 'text': corpus_map.get(fid, "")} for fid in fused_ids[:config['pipeline_params']['rerank_top_k']]]
            reranked_results = reranker.rerank(question_text, candidates_for_rerank)
            final_retrieved_articles = reranked_results [:profile['threshold']]
        else:
            final_ids = fused_ids[:profile['threshold']]
            final_retrieved_articles = []
            for full_id in final_ids:
                try:
                    law_id, article_id = full_id.rsplit('_', 1)
                    final_retrieved_articles.append({'law_id':law_id, 'article_id':article_id})
                except ValueError: continue
                
        # Xây dựng ngữ cảnh từ các điều luật đã truy xuất
        context_texts = []
        for article_info in final_retrieved_articles:
            # Fetch full_id to search in corpus_map
            full_id = f"{article_info['law_id']}_{article_info['article_id']}"
            if full_id in corpus_map:
                context_texts.append(corpus_map[full_id])
        context_string = "\n\n".join(context_texts)
        
        # --- Giai đoạn 2: Generation (Sinh câu trả lời using LangChain) ---
        chain = qa_chains.get(question_type)
        if not chain:
            logging.warning(f"No specific chain for type '{question_type}', using 'Tự luận' chain.")
            chain = qa_chains['Tự luận']
        
        # Prepare input for Chain
        chain_input = {"context": context_string}
        if question_type == "Đúng/Sai":
            statement = re.sub(r'[,]?\s+đúng\s+hay\s+sai\??$', '', question_text, flags=re.IGNORECASE).strip()
            chain_input["statement"] = statement
        elif question_type == "Trắc nghiệm":
            choices_text = "\n".join([f"{key}. {value}" for key, value in item['choices'].items()])
            chain_input["question"] = question_text
            chain_input["choices"] = choices_text
        else: # Tự luận
            chain_input["question"] = question_text
        
        # Call chain and post preprocess
        raw_answer = chain.invoke(chain_input)['text']
        final_answer = post_process_answer(raw_answer, question_type)
                 
        # Submission for Task 2
        submission_item = {
            "question_id": question_id,
            "answer": final_answer
        }
        submission_results.append(submission_item)
    
    # Write file submission
    output_file = profile['output_file']
    logging.info(f"[4/4] Write submission file to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.jump(submission_results, f, ensure_ascii=False, indent=4)
        
    logging.info(f"Task 2 Submission file '{output_file}' created sucessfully")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission for ALQAC 2025 (Task 2)")
    parser.add_argument(
        'profile_name',
        type=str,
        help="The name of the submission profile defined in config.yaml (e.g., rerank_optimal)."
    )
    args = parser.parse_args()
    main(args)