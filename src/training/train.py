import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import argparse
from src.data.generator import RTNGenerator
from src.data.dataset import create_rtn_dataloader
from src.models.transformer import RTNDualHeadTransformer

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 1. Initialize Generator and DataLoader
    generator = RTNGenerator(seq_length=args.seq_length)
    dataloader = create_rtn_dataloader(
        generator, 
        num_samples=args.num_samples, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # 2. Initialize Model
    model = RTNDualHeadTransformer(
        seq_length=args.seq_length,
        in_channels=1,
        d_model=64,
        n_heads=4,
        num_layers=3
    ).to(device)
    
    # RTX A5000 Graph Compilation
    # PyTorch 2.x feature for extreme speedup
    print("Compiling model graph via torch.compile()...")
    model = torch.compile(model)
    
    # 3. Optimizers & Scaler for Mixed Precision
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    
    # Loss functions
    criterion_seq = nn.CrossEntropyLoss()
    criterion_params = nn.MSELoss()
    
    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_seq_loss = 0.0
        total_param_loss = 0.0
        
        for batch_idx, (x, y_seq, y_params) in enumerate(dataloader):
            x = x.to(device)
            y_seq = y_seq.to(device)
            y_params = y_params.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP Forward Pass
            with autocast():
                seq_logits, params_pred = model(x)
                
                # Reshape logits for CrossEntropy: [Batch, Channels, SeqLen]
                seq_logits = seq_logits.transpose(1, 2)
                
                loss_seq = criterion_seq(seq_logits, y_seq)
                
                # Parameter scaling (log scale might be better, but standard MSE for now)
                # Note: tau_c and tau_e are small (1e-7), so scale them up for stable gradients
                loss_param = criterion_params(params_pred * 1e7, y_params * 1e7)
                
                loss = loss_seq + 0.1 * loss_param # Weighting factor
                
            # AMP Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_seq_loss += loss_seq.item()
            total_param_loss += loss_param.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Total Loss: {loss.item():.4f} | Seq Loss: {loss_seq.item():.4f} | Param Loss: {loss_param.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} ---")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, os.path.join(args.save_dir, f"rtn_transformer_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    train(args)
