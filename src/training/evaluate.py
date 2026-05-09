"""
evaluate.py – per-sample visual report for the RTN Transformer v2.
Usage:
    python -m src.training.evaluate --checkpoint checkpoints/rtn_transformer_best.pt
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
from scipy.signal import medfilt

from src.data.generator import RTNGenerator
from src.data.dataset import FinFETRTNDataset
from src.models.transformer import RTNDualHeadTransformer
from torch.utils.data import DataLoader


DARK = dict(bg='#0D1117', panel='#161B22', text='#E6EDF3',
            grid='#30363D', green='#3FB950', red='#F85149',
            blue='#58A6FF', orange='#D29922')


def _ax(ax, title):
    ax.set_facecolor(DARK['panel'])
    ax.tick_params(colors=DARK['text'], labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(DARK['grid'])
    ax.grid(True, color=DARK['grid'], lw=0.4, ls='--', alpha=0.55)
    ax.set_title(title, color=DARK['text'], fontsize=9.5, fontweight='bold', pad=5)


def decode_with_hysteresis(prob1, hi=0.65, lo=0.35):
    """Schmitt-trigger on soft probability → zero-chatter hard output."""
    out   = np.zeros_like(prob1)
    state = int(prob1[0] > 0.5)
    for i, p in enumerate(prob1):
        if   state == 0 and p > hi: state = 1
        elif state == 1 and p < lo: state = 0
        out[i] = float(state)
    return out


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: Expected Calibration Error (LLM-TSFD metric)"""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return ece, prob_pred, prob_true, bin_total[nonzero]


def evaluate(checkpoint_path, save_dir, num_samples=5, seq_length=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on: {device}")

    generator = RTNGenerator(seq_length=seq_length)
    data_list = generator.generate_batch_multiprocess(
        num_samples, num_workers=min(num_samples, 32))
    dataset   = FinFETRTNDataset(data_list)
    loader    = DataLoader(dataset, batch_size=1, shuffle=False)

    model = RTNDualHeadTransformer(
        seq_length=seq_length, in_channels=1,
        d_model=128, n_heads=8, num_layers=4).to(device)

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model_state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (x, y_seq, y_params) in enumerate(loader):
            x = x.to(device)

            logits, params_pred = model(x)          # [1,L,2], [1,2]

            # Soft probability (temp=0.4 → sharp but not hard)
            probs   = torch.softmax(logits / 0.4, dim=-1)
            prob1   = probs[0, :seq_length, 1].cpu().numpy()

            pred_hard = decode_with_hysteresis(prob1)
            true_seq  = y_seq.squeeze().numpy().astype(np.float32)
            noisy_sig = x.squeeze().cpu().numpy()

            # ── Parameter decoding (log10 → seconds) ─────────────────────────
            pred_log  = params_pred.squeeze().cpu().numpy()
            pred_tau  = 10.0 ** pred_log                        # seconds
            true_tau  = y_params.squeeze().numpy()              # raw seconds

            acc = np.mean(pred_hard == true_seq) * 100.0
            ece_val, prob_pred, prob_true, bin_counts = expected_calibration_error(true_seq, prob1)
            err_c = abs(pred_tau[0] - true_tau[0]) / true_tau[0] * 100
            err_e = abs(pred_tau[1] - true_tau[1]) / true_tau[1] * 100
            print(f"  Sample {i+1}: Acc={acc:.1f}%  ECE={ece_val:.3f}  "
                  f"τ_c err={err_c:.1f}%  τ_e err={err_e:.1f}%")

            t = np.arange(seq_length)

            fig = plt.figure(figsize=(15, 16), facecolor=DARK['bg'])
            gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.32)

            # ── Plot 1: raw overlay ───────────────────────────────────────────
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(t, noisy_sig, color='#555F6D', alpha=0.5, lw=0.7,
                     label='Noisy Input')
            ax1.plot(t, true_seq + 1.5, color=DARK['green'], lw=1.8,
                     label='True RTN State (offset +1.5)')
            ax1.plot(t, pred_hard + 3.0, color=DARK['red'], lw=1.8,
                     label='AI Denoised (offset +3.0)')
            ax1.legend(fontsize=8, facecolor=DARK['panel'],
                       labelcolor=DARK['text'], loc='upper right')
            ax1.set_xlim(0, seq_length)
            _ax(ax1, f"Signal Overview  |  State Accuracy: {acc:.1f}%  |  ECE: {ece_val:.3f}")

            # ── Plot 2: probability ───────────────────────────────────────────
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.fill_between(t, prob1, color=DARK['blue'], alpha=0.2)
            ax2.plot(t, prob1, color=DARK['blue'], lw=1.2, label='P(state=1)')
            ax2.plot(t, true_seq * 0.85, color=DARK['green'], lw=1.2,
                     alpha=0.7, label='True (scaled)')
            ax2.axhline(0.65, color=DARK['red'],    lw=0.8, ls='--', label='Hyst. Hi')
            ax2.axhline(0.35, color=DARK['orange'], lw=0.8, ls='--', label='Hyst. Lo')
            ax2.set_ylim(-0.05, 1.15)
            ax2.legend(fontsize=7.5, facecolor=DARK['panel'], labelcolor=DARK['text'])
            _ax(ax2, "Posterior Probability P(state=1)")

            # ── Plot 3: state recovery zoom ───────────────────────────────────
            ax3 = fig.add_subplot(gs[1, 1])
            z = slice(seq_length // 3, seq_length // 3 + 256)
            ax3.plot(t[z], true_seq[z],  color=DARK['green'], lw=2.0,
                     label='True State')
            ax3.plot(t[z], pred_hard[z], color=DARK['red'], lw=1.8,
                     ls='--', label='AI Denoised')
            ax3.plot(t[z], noisy_sig[z], color='#555F6D', alpha=0.4, lw=0.7,
                     label='Noisy')
            ax3.legend(fontsize=7.5, facecolor=DARK['panel'], labelcolor=DARK['text'])
            _ax(ax3, "State Recovery — Zoom (256 samples)")

            # ── Plot 4: error signal ──────────────────────────────────────────
            ax4 = fig.add_subplot(gs[2, 0])
            error = np.abs(pred_hard - true_seq)
            ax4.fill_between(t, error, color=DARK['red'], alpha=0.5, label='|error|')
            ax4.set_ylim(-0.05, 1.2)
            ax4.legend(fontsize=8, facecolor=DARK['panel'], labelcolor=DARK['text'])
            _ax(ax4, f"Error Signal  (BER = {1-acc/100:.4f})")

            # ── Plot 5: parameter bars ────────────────────────────────────────
            ax5 = fig.add_subplot(gs[2, 1])
            labels = [r'$\tau_c$', r'$\tau_e$']
            x_pos  = [0.0, 1.0]
            w      = 0.28
            true_us = true_tau * 1e6
            pred_us = pred_tau * 1e6
            ax5.bar([p - w/2 for p in x_pos], true_us, width=w,
                    color=DARK['green'], alpha=0.85, label='True (µs)')
            ax5.bar([p + w/2 for p in x_pos], pred_us, width=w,
                    color=DARK['blue'],  alpha=0.85, label='Predicted (µs)')
            ax5.set_xticks(x_pos); ax5.set_xticklabels(labels, fontsize=10)
            ax5.set_ylabel("Time (µs)", color=DARK['text'])
            ax5.legend(fontsize=8, facecolor=DARK['panel'], labelcolor=DARK['text'])
            errs = f"τ_c err={err_c:.1f}%  τ_e err={err_e:.1f}%"
            _ax(ax5, f"Parameter Regression  [{errs}]")

            # ── Plot 6: Time-Lag Plot (TLP) ───────────────────────────────────
            ax6 = fig.add_subplot(gs[3, 0])
            lag = 1
            ax6.scatter(noisy_sig[:-lag], noisy_sig[lag:], s=2, alpha=0.15, color=DARK['blue'])
            _ax(ax6, f"Time-Lag Plot (TLP) lag={lag} smp")
            ax6.set_xlabel("I(t)")
            ax6.set_ylabel("I(t + lag)")

            # ── Plot 7: ECE Reliability Diagram ───────────────────────────────
            ax7 = fig.add_subplot(gs[3, 1])
            ax7.plot([0, 1], [0, 1], ls='--', color=DARK['grid'])
            ax7.plot(prob_pred, prob_true, marker='o', color=DARK['orange'], lw=1.5, label='Calibration Curve')
            ax7.set_xlabel("Mean Predicted Probability")
            ax7.set_ylabel("Fraction of Positives")
            ax7.legend(fontsize=8, facecolor=DARK['panel'], labelcolor=DARK['text'])
            _ax(ax7, f"Reliability Diagram (ECE = {ece_val:.3f})")

            for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
                ax.yaxis.label.set_color(DARK['text'])
                ax.xaxis.label.set_color(DARK['text'])

            path = os.path.join(save_dir, f'denoising_sample_{i+1}.png')
            plt.savefig(path, dpi=200, bbox_inches='tight',
                        facecolor=DARK['bg'], edgecolor='none')
            plt.close()
            print(f"  [✓] Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--save_dir",    type=str, default="results")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_length",  type=int, default=1024)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.save_dir, args.num_samples, args.seq_length)
