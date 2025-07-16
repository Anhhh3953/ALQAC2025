import logging
from tqdm import tqdm
from src.utils import load_config, setup_logging
from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor
from models.retrievers.bm25_retriever import BM25Retriever
# from models.retrievers.semantic_retriever import SemanticRetriever


def fuse_candidates(bm25_results, semantic_results):
    """
    Combine results from multiple retrievers and remove duplicates

    Args:
        bm25_results (_type_): _description_
        semantic_results (_type_): _description_
    """
    fused_dict = {}
    for result in bm25_results + semantic_results:
        # Use tuple as key to assure uniqueness
        key = (result['law_id'], result['article_id'])
        if key not in fused_dict:
            fused_dict[key] = result
    
    return list(fused_dict.values())

def rerank_candidates(candidates, final_top_n):
    # <- rewrite here
    return candidates[:final_top_n]
    
def calculate_metrics(predicted, ground_truth):
    """
    Calculates precision and recall for a single prediction.
    """
    predicted_set ={(item['law_id'], item['article_id']) for item in predicted}
    truth_set = {(item['law_id'], item['article_id']) for item in ground_truth}
    
    true_positives = len(predicted_set.intersection(truth_set))
    
    precision = true_positives/ len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = true_positives / len(truth_set) if len(truth_set) > 0 else 0.0
    
    return precision, recall

def main():
    """
    Main funtion to run the evaluation pipeline
    """
    config = load_config()
    setup_logging(config)
    
    # 1. Load Data
    data_loader = DataLoader(config)
    law_corpus = data_loader.load_law_corpus()
    train_data = data_loader.load_data(data_name='train_set')
    stopwords = data_loader.load_stopwords()
    
    # 2. Initilize Components
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
    
    # 3. Run Evaluation Loop
    total_precision = 0
    total_recal = 0
    retrieval_params = config['retrieval_params']
    logging.info('Start evaluation on training dataset')
    for item in tqdm(train_data, desc='Evaluating'):
        question = item['text']
        ground_truth = ['relevant_articles']
        
        # Step 01: Retrieve candidates from both systems
        bm25_candidates = bm25_retriever.retrieve(question, top_k=retrieval_params['bm25_top_k'])
        # semantic_candidates = semantic_retriever.retrieve(question, top_k=retrieval_params['semantic_top_k'])
        
        # Step 2: Fuse candidate (Person A)
        fused = fuse_candidates(bm25_candidates) # semantic_candidates
        
        # Step 03: Re-rank candidates (Person B)
        final_predictions = rerank_candidates(fused, final_top_n=retrieval_params['final_top_n'])
        
        # Step 04: Calculate metrics
        precision, recall = calculate_metrics(final_predictions, ground_truth)
        total_precision += precision
        total_recall += recall
        
    # --- 4. Report Results ---
    num_questions = len(train_data)
    avg_precision = total_precision / num_questions if num_questions > 0 else 0
    avg_recall = total_recall / num_questions if num_questions > 0 else 0
    
    logging.info("----------- Evaluation Finished -----------")
    logging.info(f"Total questions evaluated: {num_questions}")
    logging.info(f"Average Precision: {avg_precision:.4f}")
    logging.info(f"Average Recall: {avg_recall:.4f}")
    logging.info("-----------------------------------------")

if __name__ == "__main__":
    main()
    