# =================================================================
# PROJECT: NLP Role Recognition
# PURPOSE: NLP with build and train the model
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
MODEL_DIR = "./model"
HF_TOKEN = ""
TfIDF_LABEL_FILE = f"{MODEL_DIR}/tfidf_labels.json"
TfIDF_VECTOR_FILE = f"{MODEL_DIR}/tfidf_vectorizer.pkl"
TfIDF_MODEL_FILE = f"{MODEL_DIR}/logistic_regression_model.pkl"
PhoBERT_LABEL_ENCODER = f"{MODEL_DIR}/PhoBert_Label_Encoder.json"
BiLSTM_MODEL_FILE = f"bilstm_role_tagger.pth"
BiLSTM_CONFIG_FILE = f"bilstm_config.json"