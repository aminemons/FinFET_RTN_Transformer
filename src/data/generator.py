import numpy as np
from scipy.signal import lfilter
import multiprocessing as mp


class RTNGenerator:
    """
    RTN Signal Generator for sub-10nm FinFET traps.
    Physical basis: Son et al. (2011) – SRH statistics / capture-emission theory.

    Improvements over v1:
    - Log-uniform tau sampling → covers 3 decades, avoids bias toward long dwells
    - Variable RTN amplitude    → models trap-dependent current step ΔId
    - Variable noise std        → tests generalisation across SNR regimes
    - Multi-trap mode           → superimposes independent 2-state traps
    """

    def __init__(
        self,
        seq_length: int,
        dt: float = 1e-9,
        tau_c_range: tuple = (5e-8, 5e-5),   # wider: 3 decades
        tau_e_range: tuple = (5e-8, 5e-5),
        rc_time_constant: float = 5e-8,
        noise_std_range: tuple = (0.05, 0.20),  # variable SNR
        amplitude_range: tuple = (0.6, 1.0),    # variable RTN step size
        multi_trap: bool = False,               # overlay 2 independent traps
        n_traps: int = 2,
    ):
        self.seq_length     = seq_length
        self.dt             = dt
        self.tau_c_range    = tau_c_range
        self.tau_e_range    = tau_e_range
        self.rc             = rc_time_constant
        self.noise_std_range = noise_std_range
        self.amplitude_range = amplitude_range
        self.multi_trap     = multi_trap
        self.n_traps        = n_traps

    # ──────────────────────────────────────────────────────────────────────────
    def _single_trap(self, tau_c: float, tau_e: float, amplitude: float) -> np.ndarray:
        """Generate one binary RTN process and return the clean amplitude signal."""
        total_time = self.seq_length * self.dt
        t_cur  = 0.0
        state  = np.random.choice([0, 1])
        times, states = [], []

        while t_cur < total_time:
            times.append(t_cur)
            states.append(state)
            tau_dwell = tau_e if state == 1 else tau_c
            t_cur += np.random.exponential(tau_dwell)
            state = 1 - state

        times.append(total_time)
        states.append(state)

        # Interpolate onto discrete time grid
        t_grid = np.arange(self.seq_length) * self.dt
        signal = np.zeros(self.seq_length, dtype=np.float32)
        idx = 0
        for i, t in enumerate(t_grid):
            while idx < len(times) - 1 and times[idx + 1] <= t:
                idx += 1
            signal[i] = states[idx] * amplitude

        return signal

    # ──────────────────────────────────────────────────────────────────────────
    def generate_sample(self, seed: int = None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        # Log-uniform sampling: equal probability per decade
        log_lo_c, log_hi_c = np.log10(self.tau_c_range[0]), np.log10(self.tau_c_range[1])
        log_lo_e, log_hi_e = np.log10(self.tau_e_range[0]), np.log10(self.tau_e_range[1])
        tau_c = float(10 ** np.random.uniform(log_lo_c, log_hi_c))
        tau_e = float(10 ** np.random.uniform(log_lo_e, log_hi_e))

        amplitude  = float(np.random.uniform(*self.amplitude_range))
        noise_std  = float(np.random.uniform(*self.noise_std_range))

        if self.multi_trap:
            # Superimpose n_traps independent traps with separate taus
            clean_rtn = np.zeros(self.seq_length, dtype=np.float32)
            for _ in range(self.n_traps):
                tc_i = float(10 ** np.random.uniform(log_lo_c, log_hi_c))
                te_i = float(10 ** np.random.uniform(log_lo_e, log_hi_e))
                amp_i = float(np.random.uniform(0.3, 0.7))
                clean_rtn += self._single_trap(tc_i, te_i, amp_i)
            # Normalise to [0, 1] for consistent target space
            max_v = clean_rtn.max()
            if max_v > 0:
                clean_rtn = clean_rtn / max_v
            # Discrete state: round to nearest level
            clean_discrete = np.round(clean_rtn).astype(np.float32)
        else:
            clean_discrete = self._single_trap(tau_c, tau_e, amplitude=1.0)
            clean_rtn      = clean_discrete.copy()

        # Parasitic RC Low-Pass Filter (bilinear IIR)
        alpha      = self.dt / (self.rc + self.dt)
        filtered   = lfilter([alpha], [1.0, -(1.0 - alpha)], clean_rtn).astype(np.float32)

        # Additive White Gaussian Noise (thermal)
        noisy = filtered + np.random.normal(0.0, noise_std, self.seq_length).astype(np.float32)

        return {
            'noisy_signal': noisy,
            'clean_signal': clean_discrete,   # integer-valued for classification target
            'filtered_signal': filtered,       # RC-filtered clean (useful for regression targets)
            'tau_c': np.float32(tau_c),
            'tau_e': np.float32(tau_e),
            'amplitude': np.float32(amplitude),
            'noise_std': np.float32(noise_std),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def generate_batch_multiprocess(self, num_samples: int, num_workers: int = 32) -> list:
        seeds = np.random.randint(0, int(1e9), size=num_samples, dtype=np.int32)
        with mp.Pool(num_workers) as pool:
            results = pool.map(self.generate_sample, seeds.tolist())
        return results
