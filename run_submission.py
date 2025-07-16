import json
import logging
from tqdm import tqdm
from src.utils import load_config, setup_logging
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from models.retrievers.bm25_retriever import BM25Retriever
# from models.retrievers.semantic_retriever import SemanticRetriever
from evaluate import fuse_candidates, rerank_candidates

def main():
    """
    Main function to run the full pipeline and generate the submission file.
    """
    config = load_config()
    setup_logging(config)
    
    # --- 1. Load Data ---
    data_loader = DataLoader(config)
    law_corpus = data_loader.load_law_corpus()
    test_data = data_loader.load_data(data_name='test_set')
    stopwords = data_loader.load_stopwords()
    
    # --- 2. Initialize Components ---
    logging.info("Initializing pipeline components...")
    preprocessor = TextPreprocessor(stopwords)
    
    # Person A's component
    bm25_retriever = BM25Retriever(
        preprocessor=preprocessor,
        model_path=config['filepaths']['bm25_model_path']
    )
    bm25_retriever.fit_or_load(corpus=law_corpus)
    
    # # Person B's component (placeholder)
    # semantic_retriever = SemanticRetriever()
    # semantic_retriever.fit(law_corpus)
    
    # 3. Process Test Data and Generate Submissions
    submission_results = []
    retrieval_params = config['retrieval_params']
    
    logging.info('Processing testing questions to generate submission file')
    for item in tqdm(test_data, desc="Generating submission"):
        question_id = item['question_id']
        question_text = item['text']
        
        # Step 01: Retrieve
        bm25_candidates = bm25_retriever.retrieve(question_text, top_k=retrieval_params['bm25_top_k'])
        # semantic_candidates = semantic_retriever.retrieve(question_text, top_k=retrieval_params['semantic_top_k'])
        
        # Step 02:
        fused = fuse_candidates(bm25_candidates) # semantic_candidates
        
        # Step 3: Re-rank
        final_predictions = rerank_candidates(fused, final_top_n=retrieval_params['final_top_n'])
        
        # Step 04: Format submission
        submission_item = {
            "question_id": question_id,
            "relevant_articles": final_predictions
        }
        submission_results.append(submission_item)
        
    # --- 4. Write Submission File ---
    submission_path = config['file_paths']['submission_file']
    logging.info(f"Writing final submission file to: {submission_path}")
    with open(submission_path, 'w', encoding='utf-8') as f:
        json.dump(submission_results, f, ensure_ascii=False, indent=4)
        
    logging.info("Submission file created successfully!")

if __name__ == "__main__":
    main()