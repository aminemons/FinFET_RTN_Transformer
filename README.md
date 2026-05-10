# Physics-Grounded AI for Random Telegraph Noise (RTN) Denoising and Parameter Extraction in Nanoscale FinFETs

**Author:** Amine Allab,  Student in Micro and Nanoelectronics

---

## 🔬 Research Overview

As semiconductor devices scale down to the nanometer regime (e.g., FinFETs, GAAFETs), individual atomic-scale defects become critical performance limiters. One of the most severe manifestations of this is **Random Telegraph Noise (RTN)**—stochastic, discrete fluctuations in the drain current caused by the dynamic trapping and emission of single charge carriers at oxide defects.

Traditional signal processing techniques (such as Moving Average, Savitzky-Golay, or Wavelet filtering) and standard statistical models (like Hidden Markov Models) fundamentally struggle to recover the true physical state of the trap when the Signal-to-Noise Ratio (SNR) degrades, leading to "chatter" and false state transitions. Furthermore, extracting the underlying physics—specifically the **Capture Time ($\tau_c$)** and **Emission Time ($\tau_e$)**—from heavily obscured signals remains computationally expensive and highly inaccurate with classical methods.

This project introduces a **Dual-Head Transformer Architecture** specifically engineered for the physics of solid-state devices. By learning the temporal long-range dependencies and the log-normal distributed switching statistics of RTN, this deep learning framework achieves state-of-the-art state recovery and instantaneous physical parameter regression directly from raw, noisy time-series data.

---

## 🧠 Architecture: The Dual-Head RTN Transformer

The core of this repository is a specialized attention-based neural network designed to simultaneously solve two problems:
1. **High-Fidelity State Recovery (Classification Head):** A sequence-to-sequence mapping that filters out background thermal/flicker ($1/f$) noise to recover the discrete two-level (or multi-level) RTN signal.
2. **Physical Parameter Regression (Regression Head):** A secondary pathway that analyzes the temporal dynamics to directly predict the characteristic time constants ($\tau_c$ and $\tau_e$) of the active trap.

### Key Innovations
*   **Physics-Grounded Inductive Bias:** Unlike standard LLMs, this Transformer is tuned to expect exponential and log-normal dwell time distributions inherent to Shockley-Read-Hall (SRH) recombination kinetics.
*   **Schmitt-Trigger Soft Decision Hysteresis:** The model's posterior probability output is passed through a custom hysteresis thresholding mechanism (mimicking a Schmitt trigger) to eliminate edge-chatter, ensuring physically valid, clean state transitions.
*   **Sub-Millisecond Inference:** Optimized via ONNX export for potential real-time integration into FPGA-based testing equipment or MATLAB environments.

---

## 📊 Experimental Validation & Output Analysis

The efficacy of the model is validated against an extensive suite of classical benchmarks (HMM + Viterbi, BiLSTM, Dilated TCN, Savitzky-Golay). The standard evaluation output of this pipeline generates a comprehensive 4-panel diagnostic figure.

### 1. Real-Time Denoising Comparison (Overlay)
The top panel of our diagnostic output overlays the raw, heavily corrupted input signal with the Transformer's filtered output. While classical filters smear the transitions (destroying the temporal resolution of the trap event), the Transformer achieves perfectly sharp, instantaneous transitions that align with the quantum mechanical nature of the electron capture/emission process.

### 2. Transformer Posterior Probability (Soft Decisions)
The second panel reveals the model's internal confidence state—$P(\text{state}=1)$. We observe that the model maintains a near-zero or near-one probability with extreme certainty, only fluctuating exactly at the transition edges. This high confidence demonstrates the attention mechanism's ability to lock onto the underlying state despite massive additive Gaussian and $1/f$ noise.

### 3. State Recovery Accuracy
The third panel plots the absolute ground truth (the true physical state of the trap) against the AI's hard decision. Our evaluations consistently show near-100% accuracy, with zero false positives (chatter) during the dwell states, outperforming traditional HMMs which often fail under low SNR.

### 4. Dual-Head Physical Parameter Regression
The final panel demonstrates the regression head's output. It compares the true Capture Time ($\tau_c$) and Emission Time ($\tau_e$) against the AI's instantaneous predictions. Traditional methods require analyzing thousands of transitions to build a Time-Lag Plot (TLP) or histogram to extract these values. Our Transformer is capable of highly accurate regression (within the same order of magnitude, crucial for lifetime reliability modeling) from relatively short temporal windows.

---

## 🚀 Repository Structure & Execution

*   `src/models/transformer.py`: The core PyTorch implementation of the Dual-Head RTN Transformer.
*   `src/training/train.py`: The main training loop utilizing GPU-accelerated PyTorch.
*   `compare_methods.py`: The rigorous benchmarking suite that pits the Transformer against classical algorithms and generates the MATLAB-quality diagnostic figures described above.
*   `checkpoints/`: Contains the pre-trained weights for the RTN Transformer.

### Getting Started

1. **Environment Setup:**
   ```bash
   conda create -n rtn_env python=3.10
   conda activate rtn_env
   pip install -r requirements.txt
   ```

2. **Run Training:**
   ```bash
   python -m src.training.train --batch_size 256 --num_workers 32
   ```

3. **Generate Benchmark Analysis (The 4-Panel Plot):**
   ```bash
   python compare_methods.py --checkpoint checkpoints/rtn_transformer_epoch_50.pt
   ```

---
*This research pushes the boundaries of how we characterize and mitigate noise in next-generation nanoscale transistors, bridging the gap between deep learning and solid-state device physics.*
