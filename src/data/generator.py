import numpy as np
from scipy.signal import lfilter, firwin
from scipy.fft import rfft, irfft, rfftfreq
import multiprocessing as mp


class RTNGenerator:
    """
    RTN Signal Generator for sub-10nm FinFET traps.

    Physical grounding:
    - Son et al. (2011)     : SRH statistics, τ_c ∝ 1/n·σ·v_th, τ_e ∝ exp(E/kT)
    - Abe et al. (2011)     : τ ∈ [10µs, 20ms] log-normal; amplitude ~ Exponential
    - Talmat et al. (2012)  : 1/f + Lorentzian colored noise, not pure Gaussian
    - Wang et al. (2016)    : RC parasitic filter physically justified
    """

    def __init__(
        self,
        seq_length: int,
        dt: float = 1e-9,
        # Log-normal μ/σ for τ in log10 seconds (Son 2011, Abe 2011)
        # Mean ≈ 1µs (log10=-6), σ=1.5 decades covers Abe's 10µs–20ms range
        tau_log_mean: float = -5.5,    # log10(seconds) → ~3µs mean
        tau_log_std:  float = 1.2,     # decades of spread
        tau_min: float = 5e-8,
        tau_max: float = 5e-4,
        rc_time_constant: float = 5e-8,
        noise_std_range: tuple = (0.05, 0.20),
        one_over_f_alpha: float = 1.0,  # 1/f^alpha exponent (Talmat 2012)
        multi_trap: bool = False,
        n_traps: int = 2,
    ):
        self.seq_length      = seq_length
        self.dt              = dt
        self.tau_log_mean    = tau_log_mean
        self.tau_log_std     = tau_log_std
        self.tau_min         = tau_min
        self.tau_max         = tau_max
        self.rc              = rc_time_constant
        self.noise_std_range = noise_std_range
        self.one_over_f_alpha = one_over_f_alpha
        self.multi_trap      = multi_trap
        self.n_traps         = n_traps

    # ──────────────────────────────────────────────────────────────────────────
    def _sample_tau(self) -> float:
        """Log-normal tau sampling (Son 2011, Abe 2011) — clipped to physical range."""
        log_tau = np.random.normal(self.tau_log_mean, self.tau_log_std)
        return float(np.clip(10**log_tau, self.tau_min, self.tau_max))

    def _sample_amplitude(self) -> float:
        """Exponential amplitude distribution (Abe 2011): most traps small, few large."""
        amp = np.random.exponential(scale=0.35)  # mean=0.35, tail to 1.0+
        return float(np.clip(amp, 0.05, 1.0))

    def _colored_noise(self, noise_std: float) -> np.ndarray:
        """
        Generates 1/f^α + Lorentzian colored noise (Talmat 2012).
        More physically accurate than pure Gaussian thermal noise.
        """
        N = self.seq_length
        freqs = rfftfreq(N, d=self.dt)   # frequency axis [Hz]
        freqs[0] = 1e-10                 # avoid division by zero at DC

        # 1/f^α power spectrum
        psd_1f = 1.0 / (freqs ** self.one_over_f_alpha)

        # Lorentzian component (dominant RTN trap contribution)
        # Corner freq ~ 1/(2π·τ_eff), τ_eff ≈ 1µs typical
        tau_eff = 1e-6
        f_corner = 1.0 / (2 * np.pi * tau_eff)
        psd_lor  = 1.0 / (1.0 + (freqs / f_corner) ** 2)

        # Combine: mostly thermal (flat) + small 1/f + Lorentzian
        psd_total = noise_std**2 + 0.15 * noise_std**2 * psd_1f / psd_1f.mean() \
                  + 0.10 * noise_std**2 * psd_lor / psd_lor.mean()

        # Generate noise via spectral shaping of white noise
        white = np.random.normal(0, 1, N // 2 + 1).astype(complex)
        white.imag = np.random.normal(0, 1, N // 2 + 1)
        colored_fft = white * np.sqrt(psd_total)
        colored     = irfft(colored_fft, n=N).astype(np.float32)

        # Rescale to desired std
        colored = colored / (colored.std() + 1e-10) * noise_std
        return colored

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

        # Log-NORMAL tau sampling (Son 2011, Abe 2011) — physically correct
        tau_c = self._sample_tau()
        tau_e = self._sample_tau()

        # Exponential amplitude distribution (Abe 2011)
        amplitude = self._sample_amplitude()
        noise_std = float(np.random.uniform(*self.noise_std_range))

        if self.multi_trap:
            # Superimpose n_traps independent traps — each with its own log-normal τ (Abe 2011)
            clean_rtn = np.zeros(self.seq_length, dtype=np.float32)
            for _ in range(self.n_traps):
                tc_i  = self._sample_tau()
                te_i  = self._sample_tau()
                amp_i = self._sample_amplitude() * 0.6  # scale down per-trap
                clean_rtn += self._single_trap(tc_i, te_i, amp_i)
            max_v = clean_rtn.max()
            if max_v > 0:
                clean_rtn = clean_rtn / max_v
            clean_discrete = np.round(clean_rtn).astype(np.float32)
        else:
            clean_discrete = self._single_trap(tau_c, tau_e, amplitude=1.0)
            clean_rtn      = clean_discrete.copy()

        # Parasitic RC Low-Pass Filter (Wang 2016 / Son 2011: RC ≈ 50ns physically)
        alpha    = self.dt / (self.rc + self.dt)
        filtered = lfilter([alpha], [1.0, -(1.0 - alpha)], clean_rtn).astype(np.float32)

        # Colored 1/f + Lorentzian noise (Talmat 2012) — physically accurate
        noisy = filtered + self._colored_noise(noise_std)

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
