import torch
import matplotlib.pyplot as plt
import os
import argparse
from src.data.generator import RTNGenerator
from src.data.dataset import RTNDataset
from src.models.transformer import RTNDualHeadTransformer
from torch.utils.data import DataLoader

def evaluate(checkpoint_path, save_dir, num_samples=5, seq_length=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    # 1. Initialize Generator and DataLoader
    generator = RTNGenerator(seq_length=seq_length)
    dataset = RTNDataset(generator, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Initialize Model
    model = RTNDualHeadTransformer(
        seq_length=seq_length,
        in_channels=1,
        d_model=64,
        n_heads=4,
        num_layers=3
    ).to(device)
    
    # Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # The model was compiled during training. We need to handle the '_orig_mod.' prefix.
    state_dict = checkpoint['model_state_dict']
    uncompiled_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(uncompiled_state_dict)
    
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (x, y_seq, y_params) in enumerate(dataloader):
            x = x.to(device)
            
            # Forward pass
            seq_logits, params_pred = model(x)
            
            # Process Sequence Prediction
            # seq_logits is [1, seq_len, 2]
            probs = torch.softmax(seq_logits, dim=-1)
            pred_seq = torch.argmax(probs, dim=-1).squeeze().cpu().numpy()
            
            # Process Param Prediction
            # The model predicts (tau_c, tau_e) scaled by 1e7
            true_params = y_params.squeeze().numpy()
            pred_params = (params_pred.squeeze().cpu() / 1e7).numpy()
            
            # Raw inputs for plotting
            noisy_signal = x.squeeze().cpu().numpy()
            true_seq = y_seq.squeeze().cpu().numpy()
            
            # Plotting
            fig, axs = plt.subplots(4, 1, figsize=(12, 10))
            
            # 1. Ground Truth Binary Sequence
            axs[0].plot(true_seq, color='green', label='True Binary State (Unobservable)')
            axs[0].set_title("Ground Truth RTN State Transitions")
            axs[0].legend(loc="upper right")
            axs[0].grid(True, alpha=0.3)
            
            # 2. Noisy RC-Filtered Input Signal
            axs[1].plot(noisy_signal, color='gray', label='Noisy RC-Filtered Measurement (Model Input)')
            axs[1].set_title("Hardware Measurement Simulation")
            axs[1].legend(loc="upper right")
            axs[1].grid(True, alpha=0.3)
            
            # 3. Model Prediction
            axs[2].plot(pred_seq, color='blue', label='Transformer Denoised Output')
            axs[2].set_title("Transformer Sequence Reconstruction")
            axs[2].legend(loc="upper right")
            axs[2].grid(True, alpha=0.3)
            
            # 4. Physical Parameter Extraction
            labels = [r'$\tau_c$ (Capture Time)', r'$\tau_e$ (Emission Time)']
            x_pos = [0, 1]
            width = 0.35
            
            axs[3].bar([p - width/2 for p in x_pos], true_params, width, label='True Parameters', color='green', alpha=0.7)
            axs[3].bar([p + width/2 for p in x_pos], pred_params, width, label='Predicted Parameters', color='blue', alpha=0.7)
            axs[3].set_xticks(x_pos)
            axs[3].set_xticklabels(labels)
            axs[3].set_title("Dual-Head Physical Parameter Regression")
            axs[3].set_ylabel("Time (seconds)")
            axs[3].legend(loc="upper right")
            axs[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'denoising_sample_{i+1}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved evaluation plot to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_length", type=int, default=1024)
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.save_dir, args.num_samples, args.seq_length)
