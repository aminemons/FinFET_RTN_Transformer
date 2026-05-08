import numpy as np
from scipy.signal import lfilter
import multiprocessing as mp

class RTNGenerator:
    def __init__(self, seq_length: int, dt: float = 1e-9, tau_c_range: tuple = (1e-7, 1e-5), tau_e_range: tuple = (1e-7, 1e-5), rc_time_constant: float = 5e-8, noise_std: float = 0.1):
        """
        RTN Signal Generator simulating sub-10nm FinFET traps.
        Physical basis: Son et al. (2011) - SRH statistics and capture theory.
        """
        self.seq_length = seq_length
        self.dt = dt
        self.tau_c_range = tau_c_range
        self.tau_e_range = tau_e_range
        self.rc = rc_time_constant
        self.noise_std = noise_std

    def generate_sample(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            
        # Uniformly sample trap constants (tau_c, tau_e)
        tau_c = np.random.uniform(*self.tau_c_range)
        tau_e = np.random.uniform(*self.tau_e_range)
        
        total_time = self.seq_length * self.dt
        t_current = 0.0
        state = np.random.choice([0, 1])
        
        times = []
        states = []
        
        # Continuous-time Markov process via exponential dwell times
        while t_current < total_time:
            times.append(t_current)
            states.append(state)
            
            tau_current = tau_e if state == 1 else tau_c
            dwell_time = np.random.exponential(tau_current)
            
            t_current += dwell_time
            state = 1 - state
            
        times.append(total_time)
        states.append(state)
        
        # Interpolate states onto discrete time grid
        t_grid = np.arange(0, total_time, self.dt)
        clean_rtn = np.zeros(self.seq_length, dtype=np.float32)
        
        idx = 0
        for i, t in enumerate(t_grid):
            while idx < len(times) - 1 and times[idx + 1] <= t:
                idx += 1
            if i < self.seq_length:
                clean_rtn[i] = states[idx]
            
        # Parasitic RC Low-Pass Filter (Bilinear transform / IIR approach)
        # Models interconnected degradation: y(t) = x(t) * e^(-t/RC)
        alpha = self.dt / (self.rc + self.dt)
        b = [alpha]
        a = [1.0, -(1.0 - alpha)]
        filtered_rtn = lfilter(b, a, clean_rtn).astype(np.float32)
        
        # Superimpose White Gaussian Noise (Thermal)
        noisy_rtn = filtered_rtn + np.random.normal(0, self.noise_std, self.seq_length).astype(np.float32)
        
        return {
            'noisy_signal': noisy_rtn,
            'clean_signal': clean_rtn,
            'tau_c': np.float32(tau_c),
            'tau_e': np.float32(tau_e)
        }

    def generate_batch_multiprocess(self, num_samples: int, num_workers: int = 32):
        seeds = np.random.randint(0, int(1e9), size=num_samples, dtype=np.int32)
        with mp.Pool(num_workers) as pool:
            results = pool.map(self.generate_sample, seeds)
        return results
