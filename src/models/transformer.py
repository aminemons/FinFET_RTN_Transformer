import torch
import torch.nn as nn
from src.models.fmla import FMLA, AbsolutePositionalEncoding
from src.models.feature_ext import StatisticalMomentExtractor

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = FMLA(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class RTNDualHeadTransformer(nn.Module):
    """
    Time-Series Transformer Neural Network.
    Input: Raw noisy RC-delayed signal.
    Head 1 (Seq2Seq): Clean denoised discrete logic state sequence (0/1).
    Head 2 (Regression): Physical trap parameters (tau_c, tau_e).
    """
    def __init__(self, seq_length: int, in_channels: int = 1, d_model: int = 64, n_heads: int = 4, num_layers: int = 3, extract_window: int = 16):
        super().__init__()
        self.seq_length = seq_length
        self.extract_window = extract_window
        
        # Allegro et al. Feature Extractor
        self.feature_extractor = StatisticalMomentExtractor(window_size=extract_window, stride=1)
        feature_dim = in_channels * 4
        
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Alioghli et al. Absolute PE
        self.pos_encoder = AbsolutePositionalEncoding(d_model, max_len=seq_length)
        
        # Transformer Layers using FMLA (O(N) Complexity)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads) for _ in range(num_layers)
        ])
        
        # Head 1: Seq2Seq State Reconstruction
        # Upsample back to original sequence length if necessary.
        # Since stride=1, sequence length is slightly reduced (SeqLen - Window + 1)
        # We'll use a Conv1D to map back or simply pad.
        self.head1_upsample = nn.ConvTranspose1d(
            in_channels=d_model, out_channels=d_model, 
            kernel_size=extract_window, stride=1
        )
        self.head1_classifier = nn.Linear(d_model, 2) # Binary classification (0/1)
        
        # Head 2: Parameter Regression
        self.head2_pool = nn.AdaptiveAvgPool1d(1)
        self.head2_regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 2) # tau_c, tau_e
        )

    def forward(self, x):
        # x shape: [Batch, Channels, SeqLen]
        
        # Feature Extraction
        # Output: [Batch, NumWindows, FeatureDim]
        features = self.feature_extractor(x)
        
        # Project to d_model
        # Output: [Batch, NumWindows, d_model]
        h = self.input_proj(features)
        
        # Positional Encoding
        h = self.pos_encoder(h)
        
        # Transformer Blocks
        for layer in self.layers:
            h = layer(h)
            
        # --- HEAD 1: Seq2Seq ---
        # Reshape for ConvTranspose1D: [Batch, d_model, NumWindows]
        h_conv = h.transpose(1, 2)
        h_upsampled = self.head1_upsample(h_conv) # [Batch, d_model, SeqLen]
        h_upsampled = h_upsampled.transpose(1, 2) # [Batch, SeqLen, d_model]
        
        # Classification logits
        seq_logits = self.head1_classifier(h_upsampled) # [Batch, SeqLen, 2]
        
        # --- HEAD 2: Regression ---
        # Global Average Pooling: [Batch, d_model]
        h_pooled = self.head2_pool(h_conv).squeeze(-1)
        params_pred = self.head2_regressor(h_pooled) # [Batch, 2]
        
        return seq_logits, params_pred
