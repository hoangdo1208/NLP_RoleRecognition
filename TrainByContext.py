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
from torch.utils.data import DataLoader, TensorDataset
import NLPRoleRecognitionCommon

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
        self.common = NLPRoleRecognitionCommon.NLPRoleRecognitionCommon()

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
        batch_size = len(conversations)
        max_len = max(len(conv["embeddings"]) for conv in conversations)

        # Pre-allocate zero tensors
        batch_embeddings = torch.zeros(batch_size, max_len, hidden_size)
        batch_labels = torch.full(
            (batch_size, max_len), self.PAD_LABEL, dtype=torch.long
        )

        for i, conv in enumerate(conversations):
            embeddings = conv["embeddings"]

            # Ensure embeddings are a Torch Tensor
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings)

            length = len(embeddings)
            batch_embeddings[i, :length] = embeddings
            batch_labels[i, :length] = torch.tensor(
                conv["labels"], dtype=torch.long
            )

        return batch_embeddings, batch_labels

    # =================================================================
    # train the BiLSTM model for role tagging using the embeddings and labels
    # =================================================================
    def trainModel(self, path: str, num_train_epochs: int = 5, batch_size: int = 8):
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        if path and path.strip():
            df = pd.read_csv(path)
        else:
            df = self.normalizeData.loadData()

        # Apply segmentation and label encoding
        df["utterance_seg"] = df["utterance"].apply(self.segmentText)
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["role"])
        print(f"Number of labels: {len(le.classes_)}, Classes: {le.classes_}")

        # Split data into train and test sets based on conversation ID
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df["conv_id"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Load tokenizer and backbone encoder model
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, use_fast=False
        )  # PhoBert uses a slow tokenizer
        encoder = AutoModel.from_pretrained(self.MODEL_NAME).to(device)
        hidden_size = encoder.config.hidden_size

        # Build conversation contexts and extract sequence embeddings
        train_conversations = self.encodeUtterances(
            self.buildConversationContext(train_df), tokenizer, encoder, device
        )
        test_conversations = self.encodeUtterances(
            self.buildConversationContext(test_df), tokenizer, encoder, device
        )

        # Collate embeddings and labels
        train_embeddings, train_labels = self.collateConversation(
            train_conversations, hidden_size
        )
        test_embeddings, test_labels = self.collateConversation(
            test_conversations, hidden_size
        )

        # Create PyTorch DataLoaders to handle batching properly
        train_dataset = TensorDataset(train_embeddings, train_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        test_dataset = TensorDataset(test_embeddings, test_labels)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        # Initialize BiLSTM role tagger model
        # Note: Use self.LSTM_HIDDEN_SIZE (or self.LSM_HIDDEN_SIZE if that is your exact property name)
        lstm_hidden_size = getattr(
            self, "LSTM_HIDDEN_SIZE", getattr(self, "LSM_HIDDEN_SIZE", 128)
        )
        model = BiLSTMRoleTagger.BiLSTMRoleTagger(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_labels=len(le.classes_),
        ) #.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.PAD_LABEL
        )  # Ignore padding labels in loss computation

        # Training loop using mini-batches
        model.train()
        for epoch in range(num_train_epochs):
            total_loss = 0.0
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                logits = model(batch_embeddings)

                # Flatten batch and sequence dimensions for CrossEntropyLoss
                loss = criterion(
                    logits.view(-1, logits.size(-1)), batch_labels.view(-1)
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(
                f"Epoch [{epoch + 1}/{num_train_epochs}], Train Loss: {avg_loss:.4f}"
            )

        # Evaluation loop
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_embeddings, batch_labels in test_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_embeddings)
                predictions = torch.argmax(logits, dim=-1)

                # Filter out padded tokens
                mask = batch_labels != self.PAD_LABEL
                all_preds.extend(predictions[mask].cpu().numpy())
                all_targets.extend(batch_labels[mask].cpu().numpy())

        # Fix: Correctly closed parentheses on classification_report call
        print("\nClassification Report:")
        print(
            classification_report(
                all_targets,
                all_preds,
                target_names=le.classes_,
                labels=range(len(le.classes_)),
                zero_division=0,
            )
        )

        # Save model checkpoint and metadata configuration
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(self.SAVE_DIR, constants.BiLSTM_MODEL_FILE),
        )

        config = {
            "model_name": self.MODEL_NAME,
            "hidden_size": hidden_size,
            "lsm_hidden_size": lstm_hidden_size,
            "labels": le.classes_.tolist(),
        }

        with open(
            os.path.join(self.SAVE_DIR, constants.BiLSTM_CONFIG_FILE),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        print(f"Model and configuration saved to {self.SAVE_DIR}")

    # =================================================================
    # Train the BiLSTM model for role tagging using the embeddings and labels
    # =================================================================
    def train(self):
        self.trainModel("", 5, 8)

    # =================================================================
    # Predict role labels for new conversation data from a .csv or .txt file.
    # Args:
    #     conversation_file (str): Path to a .csv or .txt file containing conversation data.
    # Returns:
    #     pd.DataFrame: DataFrame populated with utterances and predicted 'pred_label_id'
    #     and 'pred_label'.
    # =================================================================
    def predict(self, conversation_file: str) -> pd.DataFrame:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load data
        if not isinstance(conversation_file, str) or not os.path.exists(
            conversation_file
        ):
            raise FileNotFoundError(f"File not found: {conversation_file}")

        ext = os.path.splitext(conversation_file)[1].lower()
        if ext == ".csv":
            df = self.common.loadConversation(conversation_file)
        elif ext == ".txt":
            df = self.common.loadConversation(conversation_file)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Expected .csv or .txt file."
            )

        # 2. Add dummy label column for context builder compatibility
        pad_label = getattr(self, "PAD_LABEL", -1)
        if "label" not in df.columns:
            df["label"] = pad_label

        # 3. Load saved configuration & label map
        config_path = os.path.join(self.SAVE_DIR, constants.BiLSTM_CONFIG_FILE)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        label_classes = config["labels"]
        hidden_size = config["hidden_size"]
        lstm_hidden_size = config.get("lsm_hidden_size", 128)

        # 4. Reconstruct model and load parameters
        model_path = os.path.join(self.SAVE_DIR, constants.BiLSTM_MODEL_FILE)
        model = BiLSTMRoleTagger.BiLSTMRoleTagger(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_labels=len(label_classes),
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 5. Extract features using Transformer backbone
        df["utterance_seg"] = df["utterance"].apply(self.segmentText)

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], use_fast=False
        )
        encoder = AutoModel.from_pretrained(config["model_name"]).to(device)
        encoder.eval()

        with torch.no_grad():
            conversations = self.encodeUtterances(
                self.buildConversationContext(df), tokenizer, encoder, device
            )

        # Clean up encoder to free GPU memory
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 6. Collate and predict
        embeddings, _ = self.collateConversation(conversations, hidden_size)
        embeddings = embeddings.to(device)

        with torch.no_grad():
            logits = model(embeddings)
            predictions = torch.argmax(logits, dim=-1)

        # 7. Map numerical predictions back to role string names
        flat_preds = []
        for i, conv in enumerate(conversations):
            # Retrieve utterance sequence length safely across potential dictionary keys
            if "utterance_embeddings" in conv:
                seq_len = len(conv["utterance_embeddings"])
            elif "embeddings" in conv:
                seq_len = len(conv["embeddings"])
            elif "utterances" in conv:
                seq_len = len(conv["utterances"])
            else:
                # Fallback to checking length of input tensor/list directly
                seq_len = len(conv)

            conv_preds = predictions[i, :seq_len].cpu().numpy()

            for pred_id in conv_preds:
                flat_preds.append({
                    "pred_label_id": int(pred_id),
                    "predicted_role": label_classes[pred_id],
                })

        pred_df = pd.DataFrame(flat_preds)
        df["pred_label_id"] = pred_df["pred_label_id"]
        df["predicted_role"] = pred_df["predicted_role"]

        return df