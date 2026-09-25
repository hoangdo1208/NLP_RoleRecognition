# =================================================================
# PROJECT: NLP Role Recognition
# PURPOSE: NLP with build and train the model
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
import json
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

try:
    from underthesea import word_tokenize
    def vi_tokenize(text):
        return word_tokenize(text, format="text").split()
except ImportError:
    # If underthesea is not installed, use a simple tokenizer instead
    def vi_tokenize(text):
        return text.lower().split()

# =================================================================
# NLP Role Recognition
# =================================================================
class Tfidf:
    SAVE_DIR = "./models"

    # =================================================================
    # Load data from CSV file
    # =================================================================
    def loadData(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    # =================================================================
    # Split data into training and testing sets based on conversation ID
    # =================================================================
    def splitByConversation(self, df: pd.DataFrame, test_size=0.3, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame]:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df['conv_id']))
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        return train_df, test_df

    # =================================================================
    # Train a TF-IDF model and logistic regression classifier
    # =================================================================
    def trainModel(self, path: str) -> tuple[TfidfVectorizer, LogisticRegression]:
        # load data from CSV file
        df = self.loadData(path)

        # Split data into training and testing sets based on conversation ID
        train_df, test_df = self.splitByConversation(df)
        
        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer(tokenizer=vi_tokenize, ngram_range=(1, 2), min_df=1)
        X_train = vectorizer.fit_transform(train_df['utterance'])
        y_train = train_df['role']
        X_test = vectorizer.transform(test_df['utterance'])

        # Train a logistic regression model
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)

        print("Classification Report:")
        print(classification_report(test_df['role'], y_pred))
        print("Confusion Matrix:")
        labels = sorted(test_df['role'].unique())
        print("Labels:", labels)
        print(confusion_matrix(test_df['role'], y_pred, labels=labels))

        # Save the vectorizer and model to disk
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        joblib.dump(vectorizer, os.path.join(self.SAVE_DIR, "vectorizer.joblib"))
        joblib.dump(model, os.path.join(self.SAVE_DIR, "model.joblib"))
        with open(os.path.join(self.SAVE_DIR, "labels.json"), "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
        print(f"Vectorizer and model saved to {self.SAVE_DIR}")

        return vectorizer, model

    # =================================================================
    # Train a TF-IDF model and logistic regression classifier
    # =================================================================
    def train(self):
        pass