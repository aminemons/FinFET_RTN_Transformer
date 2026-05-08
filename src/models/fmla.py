import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FMLA(nn.Module):
    """
    Flexible Multi-head Linear Attention (FMLA)
    Achieves O(N) complexity for long sequences.
    Based on Zhao et al. (2023).
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feature map phi for linear attention (e.g., elu + 1)
        self.phi = lambda x: F.elu(x) + 1.0

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        
        # Project and reshape to [Batch, Heads, SeqLen, HeadDim]
        Q = self.q_proj(q).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply non-linear feature map and cast to FP32 to prevent FP16 summation overflow!
        # Summing 1024 elements in FP16 easily exceeds the 65,504 limit if values average > 8.0.
        Q_prime = self.phi(Q).float()
        K_prime = self.phi(K).float()
        V_fp32 = V.float()
        
        # Linear Attention Core: O(N) instead of O(N^2)
        # 1. Compute Key-Value matrix: [Batch, Heads, HeadDim, HeadDim]
        KV = torch.einsum('bhnd,bhne->bhde', K_prime, V_fp32)
        
        # 2. Multiply Query by KV matrix: [Batch, Heads, SeqLen, HeadDim]
        Z = torch.einsum('bhnd,bhde->bhne', Q_prime, KV)
        
        # Normalization term (denominator)
        normalizer = torch.einsum('bhnd,bhd->bhn', Q_prime, K_prime.sum(dim=2)).unsqueeze(-1)
        
        Z = Z / (normalizer + 1e-6)
        
        # Cast back to original dtype (e.g. FP16 inside autocast)
        Z = Z.to(Q.dtype)
        
        # Reshape back to [Batch, SeqLen, d_model]
        Z = Z.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(Z)

class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute Positional Encoding.
    Superior for multivariate anomaly detection (Alioghli et al., 2024).
    """
    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, SeqLen, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
