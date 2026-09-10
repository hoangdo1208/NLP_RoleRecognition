# =================================================================
# PROJECT: Fine-tune PhoBert for Role Recognition
# PURPOSE: Fine-tune PhoBert model for role recognition tasks
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
import torch
import torch.nn as nn

# =================================================================
# Define the BiLSTM model for role tagging
# =================================================================
class BiLSTMRoleTagger:
    # =================================================================
    # Define the BiLSTM model for role tagging
    # =================================================================
    def __init__(self, input_size: int, hidden_size: int, num_labels: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # Multiply by 2 for bidirectional

    # =================================================================
    # Forward pass through the BiLSTM model
    # =================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(lstm_out)
        return logits