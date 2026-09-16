# =================================================================
# PROJECT: Fine-tune PhoBert for Role Recognition
# PURPOSE: Fine-tune PhoBert model for role recognition tasks
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
import re

# =================================================================
# Define the CheckVietnameseEnglishSentence class to check the language of a sentence
# =================================================================
class CheckVietnameseEnglishSentence:
    # Initialize a set of Vietnamese words (SIGMA_TIENG_VIET) for language checking
    SIGMA_VIETNAMESE = {
        "tôi", "uống", "nước", "ăn", "cơm", "đi", "học"
    }

    # Initialize a set of English words (SIGMA_ENGLISH) for language checking
    SIGMA_ENGLISH = {
        "i", "you", "he", "she", "it", "we", "they",
        "drink", "drinks", "drinking", "water", "eat", "eats", 
        "go", "goes", "learn", "hello", "computer", "language"
    }
    # =================================================================
    # Initialize the CheckVietnameseEnglishSentence class
    # =================================================================
    def __init__(self):
        pass

    # =================================================================
    # Load dictionary from a text file into the Sigma set (one word per line)
    # =================================================================
    def loadFromDictionaryFile(self, file_path: str, is_vietnamese: bool) -> set:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return {line.strip().lower() for line in f if line.strip()}
        except FileNotFoundError:
            return self.SIGMA_VIETNAMESE if is_vietnamese else self.SIGMA_ENGLISH

    # =================================================================
    # Check if a sentence is in Vietnamese or English based on the Sigma sets
    # =================================================================
    def classify_sentence_language(self, sentence: str):
        # Split the sentence into words, clean them, and convert to lowercase
        raw_words = sentence.split()
        words = [re.sub(r'^\W+|\W+$', '', w).lower() for w in raw_words]
        words = [w for w in words if w]  # Remove empty strings after cleaning

        if not words:
            print("-> Chuỗi rỗng.")
            return

        # Check if all words belong to the Vietnamese or English Sigma sets
        is_pure_vn = all(word in self.SIGMA_VIETNAMESE for word in words)
        is_pure_en = all(word in self.SIGMA_ENGLISH for word in words)

        # Classify the sentence based on the checks
        if is_pure_vn and is_pure_en:
            print("-> Chuỗi thuộc cả hai tập Sigma (Tiếng Việt và Tiếng Anh).")
        elif is_pure_vn:
            print("-> Chuỗi thuộc Sigma Tiếng Việt.")
        elif is_pure_en:
            print("-> Chuỗi thuộc Sigma Tiếng Anh.")
        else:
            # Identify words that do not belong to either Sigma set
            invalid_vn = [w for w in words if w not in self.SIGMA_VIETNAMESE]
            invalid_en = [w for w in words if w not in self.SIGMA_ENGLISH]

            print("-> Chuỗi KHÔNG thuộc Sigma Tiếng Việt lẫn Tiếng Anh thuần túy.")
            print(f"   + Từ không thuộc Sigma VN: {', '.join(invalid_vn)}")
            print(f"   + Từ không thuộc Sigma EN: {', '.join(invalid_en)}")

# =================================================================
# Run the CheckVietnameseEnglishSentence class with example sentences if this script is executed directly
# =================================================================
if __name__ == "__main__":
    # Example usage
    checker = CheckVietnameseEnglishSentence()

    print("--- TEST 1: Tiếng Việt chuẩn ---")
    sentence ="Tôi uống nước"
    checker.classify_sentence_language(sentence)

    print("\n--- TEST 2: Tiếng Anh chuẩn ---")
    sentence = "I drink water"
    checker.classify_sentence_language(sentence)

    print("\n--- TEST 3: Sai từ trong Tiếng Việt ---")
    sentence = "Tôi uống thủy"
    checker.classify_sentence_language(sentence)

    print("\n--- TEST 4: Pha trộn Anh - Việt ---")
    sentence = "Tôi drink nước"
    checker.classify_sentence_language(sentence)