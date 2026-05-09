import torch
import torch.nn as nn
from src.models.fmla import FMLA, AbsolutePositionalEncoding
from src.models.feature_ext import StatisticalMomentExtractor


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn  = FMLA(d_model, n_heads)
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src  = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src  = self.norm2(src + self.dropout2(src2))
        return src


class RTNDualHeadTransformer(nn.Module):
    """
    RTN Denoising Transformer — v2.

    Changes vs v1:
    - Deeper model: 4 layers, d_model=128, 8 heads → more expressive
    - Head 1 now outputs soft log-probabilities (LogSoftmax) so NLLLoss gives
      proper probabilistic training instead of hard logits with CE.
    - Regression head: predicts LOG10(tau) targets for numerical stability
      across 3 decades of tau values.
    - Added a dedicated CRF-free smoothing conv (1-D depthwise) after the
      upsampling to suppress temporal chatter without a separate post-proc step.
    - Feature extractor stride fixed to 1 for full temporal resolution.
    """

    def __init__(
        self,
        seq_length: int,
        in_channels: int  = 1,
        d_model: int      = 128,
        n_heads: int      = 8,
        num_layers: int   = 4,
        extract_window: int = 64,   # Akbar 2021: window 64-128 optimal for RTN features
        dropout: float    = 0.1,
    ):
        super().__init__()
        self.seq_length     = seq_length
        self.extract_window = extract_window

        # ── Feature extraction (stride=1 → no temporal downsampling) ─────────
        self.feature_extractor = StatisticalMomentExtractor(
            window_size=extract_window, stride=1
        )
        feature_dim = in_channels * 4   # mean, std, skew, kurt

        self.input_proj  = nn.Linear(feature_dim, d_model)
        self.pos_encoder = AbsolutePositionalEncoding(d_model, max_len=seq_length + extract_window)

        # ── Transformer stack ─────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                    dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── Head 1: Seq2Seq state reconstruction ─────────────────────────────
        # ConvTranspose1D restores original sequence length after the feature
        # extractor's border shrinkage (SeqLen → SeqLen - W + 1).
        self.head1_upsample = nn.ConvTranspose1d(
            d_model, d_model,
            kernel_size=extract_window, stride=1
        )
        # Temporal smoothing conv (depthwise) — suppresses single-sample chatter
        self.head1_smooth = nn.Conv1d(
            d_model, d_model,
            kernel_size=7, padding=3, groups=d_model, bias=False
        )
        self.head1_classifier = nn.Linear(d_model, 2)   # 2-class logits

        # ── Head 2: Physical parameter regression ─────────────────────────────
        # Target: [log10(tau_c), log10(tau_e)] — well-conditioned across decades
        self.head2_pool = nn.AdaptiveAvgPool1d(1)
        self.head2_regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),    # → [log10(tau_c), log10(tau_e)]
        )

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, x):
        # x: [B, C, L]
        features = self.feature_extractor(x)     # [B, L', C*4]
        h = self.input_proj(features)            # [B, L', d_model]
        h = self.pos_encoder(h)

        for layer in self.layers:
            h = layer(h)

        # ── Head 1 ────────────────────────────────────────────────────────────
        h_conv      = h.transpose(1, 2)                  # [B, d_model, L']
        h_up        = self.head1_upsample(h_conv)         # [B, d_model, L]
        h_up        = self.head1_smooth(h_up)             # temporal smoothing
        h_up        = h_up[:, :, :self.seq_length]        # exact crop
        h_up_t      = h_up.transpose(1, 2)               # [B, L, d_model]
        seq_logits  = self.head1_classifier(h_up_t)      # [B, L, 2]

        # ── Head 2 ────────────────────────────────────────────────────────────
        h_pooled    = self.head2_pool(h_conv).squeeze(-1)  # [B, d_model]
        params_pred = self.head2_regressor(h_pooled)       # [B, 2] log10 scale

        return seq_logits, params_pred
