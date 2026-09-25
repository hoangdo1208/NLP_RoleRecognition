# =================================================================
# PROJECT: Fine-tune PhoBert for Role Recognition
# PURPOSE: Fine-tune PhoBert model for role recognition tasks
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from underthesea import word_tokenize
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# =================================================================
# Fine-tune PhoBert for Role Recognition
# =================================================================
class FinetunePhoBert:
    # define model name 
    MODEL_NAME = "vinai/phobert-base"

    # =================================================================
    # Defaul constructor to initialize the model and tokenizer
    # =================================================================
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    # =================================================================
    # Use underthesea for Vietnamese word segmentation
    # =================================================================
    def segmentText(self, text: str) -> str:
        # Use underthesea for Vietnamese word segmentation
        return word_tokenize(text, format="text")

    # =================================================================
    # Load data from CSV file
    # =================================================================
    def loadData(self, path: str) -> pd.DataFrame:
        # load data from CSV file
        df = pd.read_csv(path)
        return df

    # =================================================================
    # Split data into training and testing sets based on conversation ID
    # =================================================================
    def splitByConversation(self, df: pd.DataFrame, test_size=0.3, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    
        # apply segmentation to the utterance column
        df['utterance_seg'] = df['utterance'].apply(self.segmentText)  # Apply segmentation to the utterance column
    
        # encoding the role labels
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['role'])  # Encode the role labels

        # Split data into training and testing sets based on conversation ID
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(df, groups=df['conv_id']))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        return train_df, test_df, le

    # =================================================================
    # Compute evaluation metrics
    # =================================================================
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        return {"accuracy": accuracy, "macro_f1": f1}

    # =================================================================
    # Train the PhoBert model
    # =================================================================
    def trainModel(self, path: str, output_dir: str = "./models", num_train_epochs: int = 5, batch_size: int = 8):
#        # Load and split the data
        train_df, test_df, le = self.splitByConversation(self.loadData(path))
        num_labels = len(le.classes_)
        print(f"Number of labels: {num_labels}, Classes: {le.classes_}")

        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)  # PhoBert uses a slow tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)  # Adjust num_labels as needed

        ## Create datasets for training and evaluation
        def tokenize_function(batch):
            return tokenizer(batch["utterance_seg"], padding="max_length", truncation=True, max_length=64)

        # Create datasets for training and evaluation
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Tokenize the datasets
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            logging_steps=5,
            report_to="none",
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset.map(tokenize_function, batched=True),
            eval_dataset=test_dataset.map(tokenize_function, batched=True),
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        trainer.train()
        metrics = trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")

        # Save the model and tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "label_encoder.json"), "w") as f:
            json.dump(list(le.classes_), f, ensure_ascii=False)  # Save the label encoder classes for later use
        print(f"Model and tokenizer saved to {output_dir}")

    # =================================================================
    # Train the PhoBert model
    # =================================================================
    def train(self):
        pass