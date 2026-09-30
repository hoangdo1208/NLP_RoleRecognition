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
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from underthesea import word_tokenize
from transformers import AutoTokenizer, AutoModel
import BiLSTMRoleTagger as BiLSTMRoleTagger
import NormalizeData
import NLPRoleRecognitionConstants as constants

# =================================================================
# Process of train: utterance -> PhoBert (encoder) -> vector for each utterance -> BiLSTM -> Linear -> output (role)
# =================================================================
class TrainByContext:
    # define model name 
    MODEL_NAME = "vinai/phobert-base"
    PAD_LABEL = -100  # Label for padding tokens
    SAVE_DIR = constants.MODEL_DIR
    LSM_HIDDEN_SIZE = 128  # Hidden size of the BiLSTM layer

    # =================================================================
    # Defaul constructor
    # =================================================================
    def __init__(self):
        self.normalizeData = NormalizeData.NormalizeData()
        os.environ["HF_TOKEN"] = constants.HF_TOKEN

    # =================================================================
    # Segment text using underthesea for Vietnamese word segmentation
    # =================================================================
    def segmentText(self, text: str) -> str:
        # Use underthesea for Vietnamese word segmentation
        return word_tokenize(text, format="text")

    # =================================================================
    # Build conversation context by grouping utterances and labels by conversation ID
    # =================================================================
    def buildConversationContext(self, df: pd.DataFrame) -> pd.DataFrame:
        conversation_contexts = []
        for conv_id, group in df.sort_values(by='turn_id').groupby('conv_id'):
            conversation_contexts.append({
                'conv_id': conv_id,
                'utterance': group['utterance_seg'].tolist(),
                'labels': group['label'].tolist()
                })
        return conversation_contexts

    # =================================================================
    # Encode utterances using PhoBert encoder to get embeddings for each utterance
    # =================================================================
    @torch.no_grad()
    def encodeUtterances(self, conversation, tokenizer, encoder, device):
        encoder.eval()  # Set the encoder to evaluation mode
        for context in conversation:
            encode = tokenizer(context['utterance'], padding=True, truncation=True, return_tensors='pt', max_length=64).to(device)
            outputs = encoder(**encode).last_hidden_state  # Get the last hidden state
            mask = encode['attention_mask'].unsqueeze(-1) # Create a mask for the attention
            summed = (outputs * mask).sum(dim=1) # Sum the outputs along the sequence dimension
            counts = mask.sum(dim=1).clamp(min=1) # Avoid division by zero
            context['embeddings'] = (summed / counts).cpu().numpy()  # Average the embeddings and move to CPU
        return conversation

    # =================================================================
    # Pad the embeddings and labels to create uniform batch sizes for training
    # =================================================================
    def collateConversation(self, conversations, hidden_size):
        max_len = max(len(conv['labels']) for conv in conversations)
        batch_embeddings = torch.zeros((len(conversations), max_len, hidden_size))
        batch_labels = torch.full((len(conversations), max_len), self.PAD_LABEL, dtype=torch.long)
        for i, conv in enumerate(conversations):
            length = len(conv['labels'])
            batch_embeddings[i, :length] = conv['embeddings']
            batch_labels[i, :length] = torch.tensor(conv['labels'], dtype=torch.long)

        return batch_embeddings, batch_labels

    # =================================================================
    # train the BiLSTM model for role tagging using the embeddings and labels
    # =================================================================
    def trainModel(self, path: str, num_train_epochs: int = 5, batch_size: int = 8):
        # device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and split the data
        if path.strip():
            df = pd.read_csv(path)
        else:
            df = self.normalizeData.loadData()
        df['utterance_seg'] = df['utterance'].apply(self.segmentText)  # Apply segmentation to the utterance column
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['role'])  # Encode the role labels
        print(f"Number of labels: {len(le.classes_)}, Classes: {le.classes_}")

        # Split data into training and testing sets based on conversation ID
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['conv_id']))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast=False)  # PhoBert uses a slow tokenizer
        encoder = AutoModel.from_pretrained(self.MODEL_NAME).to(device)  # Load PhoBert encoder and move to device
        hidden_size = encoder.config.hidden_size  # Get the hidden size of the encoder

        # Build conversation context for training and testing sets
        train_conversations = self.encodeUtterances(self.buildConversationContext(train_df), tokenizer, encoder, device)
        test_conversations = self.encodeUtterances(self.buildConversationContext(test_df), tokenizer, encoder, device)

        # Build training and testing datasets by collating embeddings and labels
        train_embeddings, train_labels = self.collateConversation(train_conversations, hidden_size)
        test_embeddings, test_labels = self.collateConversation(test_conversations, hidden_size)

        # Initialize the BiLSTM model for role tagging
        model = BiLSTMRoleTagger(input_size=hidden_size, hidden_size=self.LSM_HIDDEN_SIZE, num_labels=len(le.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_LABEL)  # Ignore padding labels in loss computation

        # convert embeddings and labels to device
        train_embeddings, train_labels = train_embeddings.to(device), train_labels.to(device)
        test_embeddings, test_labels = test_embeddings.to(device), test_labels.to(device)

        for epoch in range(num_train_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(train_embeddings)
            loss = criterion(logits.view(-1, logits.size(-1)), train_labels.view(-1))
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch + 1}/{num_train_epochs}], Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            logits = model(test_embeddings)
            predictions = torch.argmax(logits, dim=-1)

        mask = test_labels != self.PAD_LABEL
        y_true = test_labels[mask].cpu().numpy()
        y_pred = predictions[mask].cpu().numpy()

        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=le.classes_), labels=range(len(le.classes_), zero_division=0))

        os.makedirs(self.SAVE_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.SAVE_DIR, constants.BiLSTM_MODEL_FILE))
        config = {
            "model_name": self.MODEL_NAME,
            "hidden_size": self.LSM_HIDDEN_SIZE,
            "lsm_hidden_size": self.LSM_HIDDEN_SIZE,
            "labels": le.classes_.tolist(),
            }
        with open(os.path.join(self.SAVE_DIR, constants.BiLSTM_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        print(f"Model and configuration saved to {self.SAVE_DIR}")

    # =================================================================
    # Train the BiLSTM model for role tagging using the embeddings and labels
    # =================================================================
    def train(self):
        self.trainModel("", 5, 8)

    # =================================================================
    # Predict the role in conversation file (.txt or .csv)
    # =================================================================
    def predict(self, conversation_file: str) -> pd.DataFrame:
        # Load vectorizer and model from file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = os.path.join(self.SAVE_DIR, constants.BiLSTM_CONFIG_FILE)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_name = config["model_name"]
        lsm_hidden_size = config["lsm_hidden_size"]
        labels = config["labels"]
        le = LabelEncoder()
        le.classes_ = np.array(labels)

        # load conversation from file (.csv or .txt)
        df = self.common.loadConversation(conversation_file)
        df["utterance_seg"] = df["utterance"].apply(self.segmentText)

        # Load Tokenizer & Encoder (PhoBERT)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        encoder = AutoModel.from_pretrained(model_name).to(device)
        encoder.eval()
        hidden_size = encoder.config.hidden_size

        # Extract embedding base on the context of the conversation
        conversations = self.encodeUtterances(
            self.buildConversationContext(df), tokenizer, encoder, device
        )
        embeddings, _ = self.collateConversation(conversations, hidden_size)
        embeddings = embeddings.to(device)

        # Init BiLSTM model & Load weights trained
        model = BiLSTMRoleTagger(
            input_size=hidden_size,
            hidden_size=lsm_hidden_size,
            num_labels=len(labels),
        ).to(device)

        model_weights_path = os.path.join(
            self.SAVE_DIR, constants.BiLSTM_MODEL_FILE
        )
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.eval()

        # Predict
        with torch.no_grad():
            logits = model(embeddings)
            predictions = torch.argmax(logits, dim=-1)

        # map to original role
        pred_indices = []
        for i, conv in enumerate(conversations):
            seq_len = len(conv["utterances"])
            conv_preds = predictions[i, :seq_len].cpu().numpy()
            pred_indices.extend(conv_preds)

        df["predicted_role"] = le.inverse_transform(pred_indices)

        # return the predict result
        return df