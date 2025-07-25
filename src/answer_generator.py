# src/answer_generator.py
import logging
import re
from typing import Dict

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def initialize_llm(self, model_path, llm_config):
    """Khởi tạo và trả về một instance của LlamaCpp LLM từ LangChain.

    Args:
        model_path (str): Đường dẫn đến file model .gguf trên Kaggle.
        llm_config (dict): Các tham số cấu hình cho LLM

    Returns:
        LlamaCpp: Instance của model LLM đã được tải
    """
    self.config = llm_config
    logging.info(f"Initialized LLM from path : {model_path}")
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=llm_config.get('n_gpu_layers', -1),
            n_ctx=llm_config.get('n_ctx', 4096),
            temperature=llm_config.get('temperature', 0.1),
            max_tokens=llm_config.get('max_tokens', 150),
            verbose=False
        )
        logging.info("LangChain LlamaCpp initialized successfully ")
        return llm
    except Exception as e:
        logging.error(f"Failed to load LLM model: {e}")
        return None
    
def create_qa_chains(llm):
    """Tạo các LLMChain khác nhau cho từng loại câu hỏi.

    Args:
        llm (LlamaCpp): Model LLM đã được khởi tạo.

    Returns:
        Dict[str, LLMChain]: Một dictionary chứa các chain cho 'Đúng/Sai', 'Trắc nghiệm', 'Tự luận'
    """
    # Template cho câu hỏi Đúng/Sai
    
    true_false_template = """
    Bạn là một trợ lý pháp lý được giao nhiệm vụ xác định tính đúng/sai của một phát biểu pháp luật, dựa trên nội dung được trích từ một điều luật cụ thể

    ### Nhiệm vụ của bạn:
    1. Đọc kỹ phần "Ngữ cảnh" bên dưới (nội dung của điều luật hoặc điều khoản pháp lý).
    2. Đọc kỹ phần "Phát biểu"
    3. Dựa vào thông tin trong "Ngữ cảnh", xác định xem "Phát biểu" là **Đúng** hay **Sai**

    **Chú ý quan trọng**:
    - Chỉ dựa trên ngữ cảnh được cung cấp, **không thêm kiến thức ngoài văn bản**
    - Chỉ trả lời bằng một từ duy nhất: **"Đúng"** hoặc **"Sai"** (không giải thích thêm)

    ---

    Ví dụ:

    Ngữ cảnh:
    > Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc nếu sử dụng trái phép chất ma túy trong thời gian cai nghiện ma túy tự nguyện.

    Phát biểu:
    > Người nghiện ma túy từ đủ 18 tuổi trở lên bị đưa vào cơ sở cai nghiện bắt buộc nếu bị phát hiện sử dụng trái phép chất ma túy trong thời gian đang cai nghiện tự nguyện.

    Trả lời: Đúng

    ---

    Bây giờ, hãy áp dụng quy trình trên với dữ liệu sau:

    --- Ngữ cảnh ---
    {context}
    --- Hết Ngữ cảnh ---

    Phát biểu: "{statement}"

    Chỉ trả lời bằng một từ duy nhất: "Đúng" hoặc "Sai".

    Câu trả lời:
    """

    true_false_prompt = PromptTemplate(template=true_false_template, input_variables=['context', 'statement'])
    
    # Template cho câu hỏi Trắc nghiệm
    multiple_choice_template = """
    Bạn là một trợ lý pháp lý. Nhiệm vụ của bạn là đọc và hiểu nội dung pháp luật được cung cấp, sau đó trả lời một câu hỏi trắc nghiệm bằng cách chọn ra một phương án đúng nhất.

    ### Hướng dẫn thực hiện:
    1. Đọc kỹ phần "Ngữ cảnh pháp lý" – đây là điều luật được trích dẫn nguyên văn.
    2. Đọc kỹ câu hỏi và 4 phương án lựa chọn A, B, C, D.
    3. Dựa *duy nhất vào nội dung của điều luật* (không dùng kiến thức ngoài) để chọn đáp án đúng nhất.
    4. Trả lời bằng *một ký tự duy nhất*: "A", "B", "C", hoặc "D". *Không giải thích*, không viết thêm bất kỳ từ nào

    ---

    *Ví dụ minh họa:*

    Ngữ cảnh pháp lý:
    Điều 29. Đơn phương chấm dứt hợp đồng làm việc của đơn vị sự nghiệp công lập  
    Đơn vị sự nghiệp công lập được đơn phương chấm dứt hợp đồng làm việc với viên chức trong trường hợp:  
    a) Viên chức có 02 năm liên tiếp bị phân loại đánh giá chất lượng ở mức không hoàn thành nhiệm vụ;  
    (Các điểm khác không liên quan đã được lược bỏ)


    Câu hỏi:
    Viên chức bị đơn vị sự nghiệp đơn phương chấm dứt hợp đồng trong trường hợp nào?


    Các lựa chọn:
    A. Viên chức có 02 năm liên tiếp bị phân loại đánh giá ở mức độ không hoàn thành nhiệm vụ  
    B. Viên chức ốm đau hoặc bị tai nạn, đang điều trị bệnh nghề nghiệp theo quyết định của cơ sở chữa bệnh  
    C. Viên chức đang nghỉ hàng năm, nghỉ về việc riêng và những trường hợp nghỉ khác được người đứng đầu đơn vị sự nghiệp công lập cho phép  
    D. Viên chức nữ đang trong thời gian có thai, nghỉ thai sản, nuôi con dưới 36 tháng tuổi

    Trả lời: A

    ---

    Bây giờ, hãy áp dụng đúng quy trình trên với câu hỏi sau:

    --- Ngữ cảnh pháp lý ---
    {context}
    --- Hết Ngữ cảnh ---

    Câu hỏi:
    "{question}"

    Các lựa chọn:
    {choices} 

    Chỉ trả lời bằng *một chữ cái duy nhất*: A, B, C hoặc D. Không viết gì thêm

    Đáp án:
    """
    
    multiple_choice_prompt = PromptTemplate(template=multiple_choice_template, input_variables=["context", "question", "choices"])
    
    # Template cho câu hỏi Tự luận
    free_text_template = """
    Bạn là một trợ lý pháp lý. Nhiệm vụ của bạn là trả lời một câu hỏi pháp luật ngắn gọn và chính xác, *chỉ dựa vào nội dung văn bản pháp luật được cung cấp*.

    ### Hướng dẫn:
    1. Đọc kỹ phần "Ngữ cảnh pháp lý" – đây là một điều luật cụ thể có liên quan đến câu hỏi
    2. Trả lời câu hỏi *ngắn gọn nhất có thể*, thường là một cụm từ, con số, mốc thời gian, v.v
    3. Tuyệt đối *không đưa thêm lý do, ví dụ, hoặc giải thích gì thêm*
    4. *Chỉ sử dụng thông tin có trong ngữ cảnh* – không suy đoán hay dùng kiến thức ngoài

    ---

    *Ví dụ minh họa từ Luật Trọng tài thương mại – Điều 32*:

    Ngữ cảnh pháp lý:
    Trong thời hạn 10 ngày kể từ ngày nhận được đơn khởi kiện, Trung tâm trọng tài phải gửi cho bị đơn bản sao đơn khởi kiện của nguyên đơn và những tài liệu kèm theo, trừ trường hợp các bên có thỏa thuận khác hoặc quy tắc tố tụng của Trung tâm trọng tài có quy định khác.


    Câu hỏi:
    Trong trường hợp các bên không có thỏa thuận khác hoặc quy tắc tố tụng của trung tâm trọng tài không có quy định khác, Trung tâm trọng tài phải gửi cho bị đơn bản sao đơn khởi kiện của nguyên đơn và những tài liệu theo quy định trong thời hạn bao lâu kể từ ngày nhận được đơn khởi kiện?


    Câu trả lời: *10 ngày*

    ---

    Bây giờ, hãy thực hiện tương tự với ngữ cảnh và câu hỏi sau:

    --- Ngữ cảnh ---
    {context}
    --- Hết Ngữ cảnh ---

    Câu hỏi:
    "{question}"

    Câu trả lời:
    """
    free_text_prompt = PromptTemplate(template=free_text_template, input_variables=["context", "question"])
    
    return {
        "Đúng/Sai": LLMChain(prompt=true_false_prompt, llm=llm),
        "Trắc nghiệm": LLMChain(prompt=multiple_choice_prompt, llm=llm),
        "Tự luận": LLMChain(prompt=free_text_prompt, llm=llm)
    }

def post_process_answer(raw_answer, question_type):
    """
    Hậu xử lý câu trả lời thô từ LLM để ra định dạng cuối cùng.
    
    Args:
        raw_answer (str): Chuỗi text trả về từ LLMChain.
        question_type (str): Loại câu hỏi.

    Returns:
        str: Câu trả lời đã được làm sạch.
    """
    answer = raw_answer.strip()
    if question_type == "Đúng/Sai":
        ans_lower = answer.lower()
        if "đúng" in ans_lower: return "Đúng"
        if "sai" in ans_lower: return "Sai"
        return "Không xác định"

    elif question_type == "Trắc nghiệm":
        match = re.search(r'[A-D]', answer, re.IGNORECASE)
        if match: return match.group(0).upper()
        return "A" # Default    
    elif question_type == "Tự luận":
        answer = re.sub(r'^(câu trả lời là|dựa trên ngữ cảnh,)\s*:\s*', '', answer, flags=re.IGNORECASE)
        if answer.endswith('.'): answer = answer[:-1]
        return answer.strip()

    return answer