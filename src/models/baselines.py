import torch
import torch.nn as nn

class BiLSTM_RTN(nn.Module):
    """
    BiLSTM baseline per Oh et al. 2020.
    Optimal config reported: 2 layers, hidden_size=128.
    """
    def __init__(self, in_channels=1, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # BiLSTM outputs 2 * hidden_size
        self.fc_state = nn.Linear(hidden_size * 2, 2)
        
        # Dual head for parameter regression (log-space tau)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_params = nn.Linear(hidden_size * 2, 2)
        
    def forward(self, x):
        # x: [batch, seq_len, in_channels]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        out, _ = self.lstm(x)  # [batch, seq_len, 2*hidden]
        
        # State classification
        logits = self.fc_state(out)  # [batch, seq_len, 2]
        
        # Parameter regression using global average pool
        pooled = self.pool(out.transpose(1, 2)).squeeze(-1)  # [batch, 2*hidden]
        params = self.fc_params(pooled)  # [batch, 2]
        
        return logits, params


class DilatedTCN_RTN(nn.Module):
    """
    Dilated TCN baseline per Yang et al. 2020.
    WaveNet-style architecture with increasing dilations.
    Receptive field > 64 needed.
    """
    def __init__(self, in_channels=1, num_channels=[32, 64, 64, 128], kernel_size=3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation // 2
            
            # Simple causal or non-causal? Yang used non-causal for offline denoising
            # padding='same' equivalent for Conv1d requires careful padding
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]
            
        self.network = nn.Sequential(*layers)
        self.fc_state = nn.Linear(num_channels[-1], 2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_params = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        # x: [batch, seq_len] or [batch, seq_len, in_channels]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, channels, seq_len]
        elif x.dim() == 3:
            x = x.transpose(1, 2)  # [batch, channels, seq_len]
            
        out = self.network(x)  # [batch, channels, seq_len]
        
        # State classification
        out_transposed = out.transpose(1, 2)  # [batch, seq_len, channels]
        logits = self.fc_state(out_transposed)
        
        # Parameter regression
        pooled = self.pool(out).squeeze(-1)
        params = self.fc_params(pooled)
        
        return logits, params
