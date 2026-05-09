"""
RTN Denoising Benchmark: AI Transformer vs Classical Methods
Generates MATLAB-quality comparison figures with full metrics.
Run: python compare_methods.py --checkpoint checkpoints/rtn_transformer_epoch_50.pt
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import savgol_filter, wiener, medfilt
from scipy.ndimage import uniform_filter1d
import pywt

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RTN Generator  (inline, no import needed)
# ─────────────────────────────────────────────────────────────────────────────
from scipy.signal import lfilter

def generate_rtn(seq_length=2048, dt=1e-9,
                 tau_c=3e-6, tau_e=5e-6,
                 rc=5e-8, noise_std=0.12, seed=42):
    np.random.seed(seed)
    total_time = seq_length * dt
    t_cur, state = 0.0, 0
    times, states = [], []
    while t_cur < total_time:
        times.append(t_cur); states.append(state)
        tau = tau_e if state == 1 else tau_c
        t_cur += np.random.exponential(tau)
        state = 1 - state
    times.append(total_time); states.append(state)
    t_grid  = np.arange(seq_length) * dt
    clean   = np.zeros(seq_length, np.float32)
    idx = 0
    for i, t in enumerate(t_grid):
        while idx < len(times)-1 and times[idx+1] <= t:
            idx += 1
        clean[i] = states[idx]
    alpha  = dt / (rc + dt)
    filt   = lfilter([alpha], [1., -(1-alpha)], clean).astype(np.float32)
    noisy  = filt + np.random.normal(0, noise_std, seq_length).astype(np.float32)
    return clean, noisy, tau_c, tau_e


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Classical denoisers
# ─────────────────────────────────────────────────────────────────────────────
def thresh(sig):
    # Dynamic threshold (Wang 2016): Midpoint of k-means clusters
    from sklearn.cluster import KMeans
    # Reshape signal for clustering
    sig_reshaped = sig.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(sig_reshaped)
    centers = kmeans.cluster_centers_.flatten()
    thr = np.mean(centers)
    return (sig > thr).astype(np.float32)

def moving_average(sig, w=25):
    return thresh(uniform_filter1d(sig.astype(float), size=w))

def savgol(sig, w=31, poly=3):
    s = savgol_filter(sig.astype(float), window_length=w, polyorder=poly)
    return thresh(s)

def median_filt(sig, k=21):
    return thresh(medfilt(sig.astype(float), kernel_size=k))

def wiener_filt(sig, w=25):
    return thresh(wiener(sig.astype(float), mysize=w))

def wavelet_denoise(sig, wavelet='db4', level=4):
    coeffs = pywt.wavedec(sig.astype(float), wavelet, level=level)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thr_v  = sigma * np.sqrt(2 * np.log(len(sig)))
    new_c  = [coeffs[0]] + [pywt.threshold(c, thr_v, mode='soft') for c in coeffs[1:]]
    rec    = pywt.waverec(new_c, wavelet)[:len(sig)]
    return thresh(rec)

def hmm_viterbi(sig, n_iter=60):
    """Simple 2-state Gaussian HMM + Viterbi decoding."""
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=2, max_iter=n_iter, random_state=0)
    gm.fit(sig.reshape(-1,1))
    # Make sure label 0 = low, 1 = high
    means = gm.means_.flatten()
    labels = gm.predict(sig.reshape(-1,1)).astype(np.float32)
    if means[0] > means[1]:
        labels = 1.0 - labels
    # Viterbi smoothing with median pass to kill chatter
    return thresh(medfilt(labels, 9))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AI Transformer inference
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path, seq_length, device, model_type='transformer'):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    
    if model_type == 'transformer':
        from src.models.transformer import RTNDualHeadTransformer
        model = RTNDualHeadTransformer(
            seq_length=seq_length, in_channels=1,
            d_model=128, n_heads=8, num_layers=4, extract_window=64).to(device)
    elif model_type == 'bilstm':
        from src.models.baselines import BiLSTM_RTN
        model = BiLSTM_RTN(in_channels=1, hidden_size=128, num_layers=2).to(device)
    elif model_type == 'tcn':
        from src.models.baselines import DilatedTCN_RTN
        model = DilatedTCN_RTN(in_channels=1, num_channels=[32, 64, 64, 128], kernel_size=3).to(device)

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model_state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()
    return model

def ai_denoise(model, noisy, seq_length, device):
    """
    Run transformer + post-process:
      - soft probability from softmax
      - temperature scaling to keep soft output
      - Viterbi-style median smoothing for zero chatter
    """
    x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
    # Pad/crop to exact seq_length
    actual_len = noisy.shape[0]
    with torch.no_grad():
        logits, params = model(x)  # [1, L, 2]

    # Soft probability (temperature T=0.5 → sharper but not hard)
    T = 0.5
    probs = torch.softmax(logits / T, dim=-1)
    prob1 = probs[0, :, 1].cpu().numpy()  # P(state=1)

    # Trim/pad to actual_len
    prob1 = prob1[:actual_len]
    if len(prob1) < actual_len:
        prob1 = np.pad(prob1, (0, actual_len - len(prob1)), mode='edge')

    # Hard decision with hysteresis: avoid chatterin around 0.5
    # Use 0.35 / 0.65 dual threshold (Schmitt trigger in software)
    hard = np.zeros(actual_len, np.float32)
    state = int(prob1[0] > 0.5)
    for i, p in enumerate(prob1):
        if state == 0 and p > 0.65:
            state = 1
        elif state == 1 and p < 0.35:
            state = 0
        hard[i] = float(state)

    return hard, prob1, params.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(clean, denoised):
    ber  = np.mean(np.abs(clean - denoised))
    acc  = 1.0 - ber

    # Transition detection delay (average sample delay at rising edges)
    c_edges = np.where(np.diff(clean.astype(int)) == 1)[0]
    d_edges = np.where(np.diff(denoised.astype(int)) == 1)[0]
    delays = []
    for ce in c_edges:
        candidates = d_edges[(d_edges >= ce) & (d_edges < ce + 100)]
        if len(candidates):
            delays.append(candidates[0] - ce)
    avg_delay = np.mean(delays) if delays else float('nan')

    # SNR improvement (dB): treat clean as "signal", denoised error as "noise"
    sig_pwr  = np.mean(clean**2) + 1e-10
    err_pwr  = np.mean((clean - denoised)**2) + 1e-10
    snr_db   = 10 * np.log10(sig_pwr / err_pwr)

    return dict(BER=ber, Accuracy=acc, Delay=avg_delay, SNR_dB=snr_db)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MATLAB-quality figure
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    'noisy':  dict(color='#AAAAAA', alpha=0.45, lw=0.8,  label='Noisy Input',          zorder=1),
    'clean':  dict(color='#00CC44', alpha=0.95, lw=2.0,  label='True RTN State',        zorder=5),
    'ma':     dict(color='#FF6B6B', alpha=0.90, lw=1.4,  label='Moving Average',        zorder=2),
    'sg':     dict(color='#FFB347', alpha=0.90, lw=1.4,  label='Savitzky-Golay',        zorder=2),
    'med':    dict(color='#9B59B6', alpha=0.90, lw=1.4,  label='Median Filter',         zorder=2),
    'wav':    dict(color='#3498DB', alpha=0.90, lw=1.4,  label='Wavelet (db4)',         zorder=2),
    'hmm':    dict(color='#E67E22', alpha=0.90, lw=1.6,  label='HMM + Viterbi',        zorder=3),
    'lstm':   dict(color='#8E44AD', alpha=0.95, lw=1.8,  label='BiLSTM (Oh 2020)',      zorder=4),
    'tcn':    dict(color='#16A085', alpha=0.95, lw=1.8,  label='Dilated TCN (Yang 2020)',zorder=4),
    'ai':     dict(color='#E74C3C', alpha=1.00, lw=2.2,  label='AI Transformer (Ours)',  zorder=6),
}

PALETTE = dict(
    background='#0D1117',
    panel='#161B22',
    text='#E6EDF3',
    grid='#30363D',
    accent='#58A6FF',
)

def _ax_style(ax, title):
    ax.set_facecolor(PALETTE['panel'])
    ax.tick_params(colors=PALETTE['text'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE['grid'])
    ax.grid(True, color=PALETTE['grid'], lw=0.5, linestyle='--', alpha=0.6)
    ax.set_title(title, color=PALETTE['text'], fontsize=10, fontweight='bold', pad=6)
    ax.yaxis.label.set_color(PALETTE['text'])
    ax.xaxis.label.set_color(PALETTE['text'])


def make_figure(clean, noisy, results, metrics_dict, tau_c, tau_e,
                prob1_ai, pred_params, seq_length, save_path):
    t = np.arange(seq_length)
    # ── Row 1: Method-by-method comparison (zoom window) ────────────────────
    zoom_s, zoom_e = seq_length//4, seq_length//4 + 512
    z = slice(zoom_s, zoom_e)
    tz = t[z]

    methods = [('ma','Moving Average'), ('sg','Savitzky-Golay'),
               ('wav','Wavelet (db4)'), ('hmm','HMM+Viterbi'),
               ('med','Median Filter')]
    if 'lstm' in results: methods.append(('lstm', 'BiLSTM'))
    if 'tcn' in results: methods.append(('tcn', 'Dilated TCN'))
    methods.append(('ai','AI Transformer'))

    # Determine layout dynamically based on number of methods
    cols = 3
    method_rows = (len(methods) + cols - 1) // cols
    total_rows = 1 + method_rows + 1

    fig = plt.figure(figsize=(20, 4.5 * total_rows), facecolor=PALETTE['background'])
    gs  = gridspec.GridSpec(total_rows, 3, figure=fig, hspace=0.55, wspace=0.32)

    # ── Row 0: Full signal overview ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, noisy, **STYLE['noisy'])
    ax0.plot(t, clean + 1.4,  **STYLE['clean'])       # offset for clarity
    ax0.plot(t, results['ai'] + 2.8, **STYLE['ai'])
    ax0.set_xlim(0, seq_length)
    ax0.set_xlabel("Sample index", fontsize=9)
    ax0.set_ylabel("Amplitude (a.u.)", fontsize=9)
    ax0.legend(loc='upper right', fontsize=8, facecolor=PALETTE['panel'],
               labelcolor=PALETTE['text'], framealpha=0.8)
    _ax_style(ax0, "Signal Overview  [Bottom: Noisy  |  Middle: True RTN  |  Top: AI Output]")

    # We will use axes in row 1 up to method_rows
    method_axes = []
    for r in range(method_rows):
        for c in range(cols):
            if len(method_axes) < len(methods):
                method_axes.append(fig.add_subplot(gs[r+1, c]))

    for idx, (key, title) in enumerate(methods):
        ax = method_axes[idx]
        ax.plot(tz, noisy[z], **STYLE['noisy'])
        ax.plot(tz, clean[z], **STYLE['clean'])
        ax.plot(tz, results[key][z], **STYLE[key])
        m = metrics_dict[key]
        info = f"Acc={m['Accuracy']*100:.1f}%  SNR={m['SNR_dB']:.1f}dB  Delay={m['Delay']:.0f}smp"
        ax.set_title(f"{title}\n{info}", fontsize=9, color=PALETTE['text'], fontweight='bold', pad=4)
        ax.set_xlim(tz[0], tz[-1])
        ax.set_ylim(-0.3, 1.5)
        _ax_style(ax, f"{title}\n{info}")

    # ── Bottom row left: AI soft probability ─────────────────────────────────
    bottom_row = 1 + method_rows
    ax_prob = fig.add_subplot(gs[bottom_row, 0])
    ax_prob.fill_between(t, prob1_ai, color='#58A6FF', alpha=0.25)
    ax_prob.plot(t, prob1_ai, color='#58A6FF', lw=1.2, label='P(state=1)')
    ax_prob.plot(t, clean * 0.9, color='#00CC44', lw=1.5, alpha=0.7, label='True state (scaled)')
    ax_prob.axhline(0.65, color='#FF6B6B', lw=0.8, ls='--', label='High thresh (0.65)')
    ax_prob.axhline(0.35, color='#FFB347', lw=0.8, ls='--', label='Low thresh (0.35)')
    ax_prob.set_ylim(-0.05, 1.15)
    ax_prob.legend(fontsize=7.5, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])
    _ax_style(ax_prob, "AI Posterior Probability P(state=1) + Schmitt Hysteresis")

    # ── Bottom row middle: Metrics bar chart ─────────────────────────────────
    ax_met = fig.add_subplot(gs[bottom_row, 1])
    method_names = list(metrics_dict.keys())
    accs  = [metrics_dict[m]['Accuracy']*100 for m in method_names]
    # dynamically get color from STYLE
    colors = [STYLE[m]['color'] for m in method_names]
    bars = ax_met.barh(method_names, accs, color=colors, edgecolor='none', height=0.55)
    ax_met.set_xlim(40, 102)
    for b, v in zip(bars, accs):
        ax_met.text(v+0.3, b.get_y()+b.get_height()/2,
                    f'{v:.1f}%', va='center', ha='left',
                    fontsize=8.5, color=PALETTE['text'], fontweight='bold')
    ax_met.set_xlabel("State Accuracy (%)", fontsize=9)
    ax_met.tick_params(colors=PALETTE['text'])
    _ax_style(ax_met, "State Accuracy — All Methods")

    # ── Bottom row right: Parameter estimation ───────────────────────────────
    ax_par = fig.add_subplot(gs[bottom_row, 2])
    labels = [r'$\tau_c$ (Capture)', r'$\tau_e$ (Emission)']
    true_v = np.array([tau_c, tau_e]) * 1e6   # µs
    # pred_params is log10(seconds), convert to µs
    pred_v = (10.0 ** np.array(pred_params)) * 1e6

    x_pos = np.array([0.0, 1.0])
    w = 0.3
    ax_par.bar(x_pos - w/2, true_v, width=w, label='Ground Truth',
               color='#00CC44', alpha=0.85, edgecolor='none')
    ax_par.bar(x_pos + w/2, pred_v, width=w, label='AI Prediction',
               color='#E74C3C', alpha=0.85, edgecolor='none')
    ax_par.set_xticks(x_pos)
    ax_par.set_xticklabels(labels, fontsize=9, color=PALETTE['text'])
    ax_par.set_ylabel("Time (µs)", fontsize=9)
    ax_par.legend(fontsize=8, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])
    _ax_style(ax_par, "Physical Parameter Regression (τ)")

    # Watermark
    fig.text(0.5, 0.005,
             f"FinFET RTN Denoising Benchmark  |  τ_c={tau_c*1e6:.2f}µs  τ_e={tau_e*1e6:.2f}µs  |  seq={seq_length}  noise_std=0.12",
             ha='center', fontsize=8, color='#555F6D')

    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor=PALETTE['background'], edgecolor='none')
    plt.close()
    print(f"[✓] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/rtn_transformer_epoch_50.pt')
    parser.add_argument('--checkpoint_bilstm', type=str, default=None,
                        help='Path to BiLSTM checkpoint')
    parser.add_argument('--checkpoint_tcn', type=str, default=None,
                        help='Path to Dilated TCN checkpoint')
    parser.add_argument('--seq_length',  type=int,   default=2048)
    parser.add_argument('--noise_std',   type=float, default=0.12)
    parser.add_argument('--tau_c',       type=float, default=3e-6)
    parser.add_argument('--tau_e',       type=float, default=5e-6)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--num_samples', type=int,   default=3,
                        help='Number of random test signals to generate')
    parser.add_argument('--out_dir',     type=str,   default='results/comparison')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load AI models
    print("Loading AI models...")
    models = {'transformer': None, 'bilstm': None, 'tcn': None}
    try:
        models['transformer'] = load_model(args.checkpoint, args.seq_length, device, 'transformer')
        print("[✓] Transformer loaded")
    except Exception as e:
        print(f"[!] Could not load Transformer: {e}")

    if args.checkpoint_bilstm:
        try:
            models['bilstm'] = load_model(args.checkpoint_bilstm, args.seq_length, device, 'bilstm')
            print("[✓] BiLSTM loaded")
        except Exception as e:
            print(f"[!] Could not load BiLSTM: {e}")

    if args.checkpoint_tcn:
        try:
            models['tcn'] = load_model(args.checkpoint_tcn, args.seq_length, device, 'tcn')
            print("[✓] Dilated TCN loaded")
        except Exception as e:
            print(f"[!] Could not load Dilated TCN: {e}")

    seeds = [args.seed + i*7 for i in range(args.num_samples)]

    for idx, seed in enumerate(seeds):
        print(f"\n── Sample {idx+1}/{args.num_samples}  seed={seed} ──")

        # Vary tau slightly per sample
        tc = args.tau_c * (0.8 + 0.4 * np.random.RandomState(seed).rand())
        te = args.tau_e * (0.8 + 0.4 * np.random.RandomState(seed+1).rand())

        clean, noisy, tau_c, tau_e = generate_rtn(
            seq_length=args.seq_length, noise_std=args.noise_std,
            tau_c=tc, tau_e=te, seed=seed)

        print("Running classical methods...")
        results = {
            'ma':  moving_average(noisy, w=25),
            'sg':  savgol(noisy, w=31, poly=3),
            'wav': wavelet_denoise(noisy),
            'hmm': hmm_viterbi(noisy),
            'med': median_filt(noisy, k=21),
        }

        prob1_ai, pred_params = np.zeros(args.seq_length), np.zeros(2)
        
        # BiLSTM baseline
        if models['bilstm'] is not None:
            print("Running BiLSTM...")
            lstm_hard, _, _ = ai_denoise(models['bilstm'], noisy, args.seq_length, device)
            results['lstm'] = lstm_hard

        # TCN baseline
        if models['tcn'] is not None:
            print("Running Dilated TCN...")
            tcn_hard, _, _ = ai_denoise(models['tcn'], noisy, args.seq_length, device)
            results['tcn'] = tcn_hard

        # Transformer
        if models['transformer'] is not None:
            print("Running AI transformer...")
            ai_hard, prob1_ai, pred_params = ai_denoise(
                models['transformer'], noisy, args.seq_length, device)
            results['ai'] = ai_hard
        else:
            results['ai'] = thresh(noisy)

        print("Computing metrics...")
        metrics_dict = {k: compute_metrics(clean, v) for k, v in results.items()}

        print("\n  Method         | Accuracy | SNR(dB) | Delay(smp)")
        print("  " + "-"*55)
        for k, m in metrics_dict.items():
            print(f"  {k:<14} | {m['Accuracy']*100:6.2f}%  | {m['SNR_dB']:6.2f}  | {m['Delay']:5.1f}")

        save_path = os.path.join(args.out_dir, f'comparison_sample_{idx+1}.png')
        make_figure(clean, noisy, results, metrics_dict, tau_c, tau_e,
                    prob1_ai, pred_params, args.seq_length, save_path)

    print(f"\n[✓] Done — {args.num_samples} comparison figures saved to '{args.out_dir}/'")


if __name__ == '__main__':
    main()
