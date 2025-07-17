# src/preprocessor.py

import re
import logging
from pyvi import ViTokenizer

class TextPreprocessor:
    """
    Đóng gói các bước tiền xử lý văn bản: chuẩn hóa, tách từ, và loại bỏ stopword.
    """
    def __init__(self, stopwords: set = None):
        """
        Khởi tạo Preprocessor.
        :param stopwords: Một set các từ dừng đã được load từ bên ngoài.
        """
        # Dependency Injection: Nhận stopwords từ bên ngoài, không tự đọc file.
        self.stopwords = stopwords if stopwords else set()
        
        # Regex để loại bỏ tất cả các ký tự không phải là chữ, số, hoặc dấu gạch dưới
        # Giữ lại dấu gạch dưới vì pyvi sử dụng nó để nối các từ ghép.
        self.punctuation_re = re.compile(r'[^\w\s_]')
        
        logging.info(f"TextPreprocessor initialized with {len(self.stopwords)} stopwords.")

    def process(self, text: str) -> list[str]:
        """
        Áp dụng toàn bộ pipeline tiền xử lý cho một đoạn văn bản.

        Các bước bao gồm:
        1. Chuyển thành chữ thường.
        2. Tách từ bằng PyVi.
        3. Loại bỏ các ký tự đặc biệt không cần thiết.
        4. Loại bỏ stopword và các token ngắn.
        5. Chuẩn hóa khoảng trắng.
        """
        if not isinstance(text, str):
            return []

        # 1. Chuyển thành chữ thường và tách từ
        # Kết quả: "luật phòng_chống ma_túy" -> "luật phòng_chống ma_túy"
        tokenized_text = ViTokenizer.tokenize(text.lower())
        
        # 2. Loại bỏ các dấu câu không mong muốn
        # Giữ lại chữ, số và dấu gạch dưới
        # Kết quả: "luật phòng_chống, ma_túy." -> "luật phòng_chống ma_túy"
        cleaned_text = self.punctuation_re.sub(' ', tokenized_text)

        # 3. Tách chuỗi thành các token và loại bỏ stopword
        # Kết quả: "luật phòng_chống ma_túy" -> ['luật', 'phòng_chống', 'ma_túy']
        tokens = cleaned_text.split()
        
        # Lọc stopwords và các token quá ngắn (thường là nhiễu do regex)
        # Sử dụng set stopwords để tra cứu nhanh hơn (O(1))
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stopwords and len(token) > 1
        ]
        
        return filtered_tokens