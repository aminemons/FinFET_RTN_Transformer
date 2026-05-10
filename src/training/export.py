import torch
import os
import argparse
from src.models.transformer import RTNDualHeadTransformer

def export_to_onnx(checkpoint_path, output_path, seq_length=1024):
    device = torch.device("cpu") # Export on CPU
    
    # 1. Initialize Model Architecture
    model = RTNDualHeadTransformer(
        seq_length=seq_length,
        in_channels=1,
        d_model=128,
        n_heads=8,
        num_layers=4
    ).to(device)
    
    # 2. Load Weights (Handle compiled models prefix '_orig_mod.')
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Strip compiled prefix if necessary
        uncompiled_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                uncompiled_state_dict[k.replace("_orig_mod.", "")] = v
            else:
                uncompiled_state_dict[k] = v
                
        model.load_state_dict(uncompiled_state_dict)
    else:
        print("Warning: No checkpoint provided, exporting untrained model architecture.")
        
    model.eval()
    
    # 3. Define Dummy Input
    # Batch size = 1, Channels = 1, Sequence Length = seq_length
    dummy_input = torch.randn(1, 1, seq_length, dtype=torch.float32).to(device)
    
    # 4. Export to ONNX with dynamic axes
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=14, # App Designer optimal opset
        do_constant_folding=True,
        input_names=['raw_signal'],
        output_names=['clean_sequence', 'physical_parameters'],
        dynamic_axes={
            'raw_signal': {0: 'batch_size', 2: 'sequence_length'},
            'clean_sequence': {0: 'batch_size', 1: 'sequence_length'},
            'physical_parameters': {0: 'batch_size'}
        }
    )
    
    print("ONNX Export Successful! Ready for MATLAB App Designer integration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="rtn_denoiser.onnx")
    parser.add_argument("--seq_length", type=int, default=1024)
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output, args.seq_length)
