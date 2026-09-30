# =================================================================
# PROJECT: Fine-tune PhoBert for Role Recognition
# PURPOSE: Fine-tune PhoBert model for role recognition tasks
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
import pandas as pd

# =================================================================
# The common utilities method
# =================================================================
class NLPRoleRecognitionCommon:
    # =================================================================
    # Load conversation from CSV or TXT file
    # =================================================================
    def loadConversation(self, conversationfile:str) -> pd.DataFrame:
        # load conversation from file
        if conversationfile.endswith('.csv'):\
            # Assumes the text is in a column named 'utterance'
            df = pd.read_csv(conversationfile)
            # Adjust column name 'utterance' to match your CSV structure
            return df['utterance'].tolist(), df
        elif conversationfile.endswith('.txt'):
            # Read text file line by line
            with open(conversationfile, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                return lines, pd.DataFrame({'utterance': lines})
        else:
            raise ValueError("Unsupported file format. Please use .csv or .txt")