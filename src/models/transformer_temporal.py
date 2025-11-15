"""Transformer Temporal Model for SafeFusion."""

import torch
import torch.nn as nn

class TransformerTemporal(nn.Module):
    """Transformer model for temporal accident prediction."""
    
    def __init__(self, input_dim=512, hidden_dim=256, num_heads=8, num_layers=6, num_classes=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x
