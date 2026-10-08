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
class BiLSTMRoleTagger(nn.Module):
    # =================================================================
    # Define the BiLSTM model for role tagging
    # =================================================================
    def __init__(self, input_size: int, hidden_size: int, num_labels: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_size * 2, num_labels)

    # =================================================================
    # Forward pass through the BiLSTM model
    # =================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits

    # =================================================================
    # Get parameter groups for different learning rates
    # =================================================================
    def get_parameter_groups(self, lr_lstm: float = 1e-3, lr_fc: float = 5e-3):
        return [
            {"params": self.lstm.parameters(), "lr": lr_lstm},
            {"params": self.fc.parameters(), "lr": lr_fc},
        ]