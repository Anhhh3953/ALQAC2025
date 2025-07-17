# src/fusers.py
from collections import defaultdict

def reciprocal_rank_fusion(results_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Kết hợp nhiều danh sách kết quả xếp hạng bằng Reciprocal Rank Fusion (RRF).
    
    :param results_lists: Một list chứa các list kết quả ID. 
                          Ví dụ: [['id1', 'id2'], ['id2', 'id3']]
    :param k: Tham số của RRF, mặc định là 60.
    :return: Một danh sách các ID đã được sắp xếp lại theo điểm RRF.
    """
    if not results_lists:
        return []

    rrf_scores = defaultdict(float)

    for results in results_lists:
        for rank, doc_id in enumerate(results):
            rrf_scores[doc_id] += 1 / (k + rank + 1)
            
    if not rrf_scores:
        return []

    sorted_fused_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_fused_results]