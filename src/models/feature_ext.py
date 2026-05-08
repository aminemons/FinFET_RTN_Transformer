import torch
import torch.nn as nn

class StatisticalMomentExtractor(nn.Module):
    """
    Extracts statistical moments (mean, std, skewness, kurtosis) from time-series windows.
    Based on Allegro et al. (2023) - Using statistical moments as features for noise event classification.
    """
    def __init__(self, window_size: int = 16, stride: int = 8):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        # Feature dimension is 4 (mean, std, skewness, kurtosis)
        self.feature_dim = 4

    def forward(self, x):
        # x shape: [Batch, Channels, SeqLen]
        # Unfold to get sliding windows: [Batch, Channels, NumWindows, WindowSize]
        windows = x.unfold(dimension=-1, size=self.window_size, step=self.stride)
        
        # Calculate Mean
        mean = windows.mean(dim=-1)
        
        # Calculate Std
        std = windows.std(dim=-1, unbiased=True) + 1e-8
        
        # Calculate Skewness
        # (E[(X - mu)^3] / sigma^3)
        diff = windows - mean.unsqueeze(-1)
        skewness = (diff ** 3).mean(dim=-1) / (std ** 3)
        skewness = torch.clamp(skewness, -50.0, 50.0) # Prevent FP16 explosion
        
        # Calculate Kurtosis
        # (E[(X - mu)^4] / sigma^4)
        kurtosis = (diff ** 4).mean(dim=-1) / (std ** 4)
        kurtosis = torch.clamp(kurtosis, -50.0, 50.0) # Prevent FP16 explosion
        
        # Stack features: [Batch, Channels, NumWindows, FeatureDim]
        features = torch.stack([mean, std, skewness, kurtosis], dim=-1)
        
        # Reshape to [Batch, NumWindows, Channels * FeatureDim]
        b, c, nw, fd = features.shape
        features = features.permute(0, 2, 1, 3).reshape(b, nw, c * fd)
        
        return features
