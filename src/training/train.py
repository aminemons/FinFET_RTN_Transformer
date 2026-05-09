import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import os
import argparse
import numpy as np
from src.data.generator import RTNGenerator
from src.data.dataset import create_rtn_dataloader
from src.models.transformer import RTNDualHeadTransformer


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    generator = RTNGenerator(
        seq_length=args.seq_length,
        multi_trap=args.multi_trap,
    )
    dataloader = create_rtn_dataloader(
        generator,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    steps_per_epoch = len(dataloader)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = RTNDualHeadTransformer(
        seq_length=args.seq_length,
        in_channels=1,
        d_model=128,
        n_heads=8,
        num_layers=4,
        dropout=0.1,
    ).to(device)

    print("Compiling model graph via torch.compile()...")
    model = torch.compile(model)

    # ── 3. Optimiser + LR schedule ────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    # ── 4. Loss functions ─────────────────────────────────────────────────────
    # Focal-weighted CE for the sequence head — handles class imbalance
    # at transitions (sparse 0→1 vs. long dwell plateaus).
    criterion_seq   = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Huber loss for regression: robust to outliers, smooth near zero
    criterion_params = nn.HuberLoss(delta=1.0)

    # ── 5. Training loop ──────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss       = 0.0
        total_seq_loss   = 0.0
        total_param_loss = 0.0

        for batch_idx, (x, y_seq, y_params) in enumerate(dataloader):
            x      = x.to(device, non_blocking=True)
            y_seq  = y_seq.to(device, non_blocking=True)

            # Regression targets: log10(tau) — well conditioned across decades
            # y_params shape [B, 2] contains raw tau values in seconds
            log_tau = torch.log10(y_params.clamp(min=1e-10)).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                seq_logits, params_pred = model(x)
                seq_logits_t = seq_logits.transpose(1, 2)   # [B, 2, L] for CE

            loss_seq   = criterion_seq(seq_logits_t.float(), y_seq)
            loss_param = criterion_params(params_pred.float(), log_tau.float())

            # Dynamic loss weighting:
            #   seq head = primary task (weight 1.0)
            #   param head = boosted (weight 3.0) — was previously underfit
            loss = loss_seq + 3.0 * loss_param

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss       += loss.item()
            total_seq_loss   += loss_seq.item()
            total_param_loss += loss_param.item()

            if batch_idx % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch+1:02d}/{args.epochs} | "
                    f"Batch {batch_idx:04d}/{steps_per_epoch} | "
                    f"Loss {loss.item():.4f}  Seq {loss_seq.item():.4f}  "
                    f"Param {loss_param.item():.4f}  LR {lr_now:.2e}"
                )

        avg_loss       = total_loss       / steps_per_epoch
        avg_seq_loss   = total_seq_loss   / steps_per_epoch
        avg_param_loss = total_param_loss / steps_per_epoch
        print(
            f"── Epoch {epoch+1:02d} done │ "
            f"Avg Loss {avg_loss:.4f}  Seq {avg_seq_loss:.4f}  Param {avg_param_loss:.4f}"
        )

        # Save every epoch + always keep best checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'config': vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"rtn_transformer_epoch_{epoch+1}.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(args.save_dir, "rtn_transformer_best.pt"))
            print(f"   [✓] New best checkpoint saved (loss={best_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RTN Denoising Transformer v2")
    parser.add_argument("--seq_length",  type=int,   default=1024)
    parser.add_argument("--num_samples", type=int,   default=200000,  help="Training set size")
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--num_workers", type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--save_dir",    type=str,   default="checkpoints")
    parser.add_argument("--multi_trap",  action="store_true",
                        help="Train on multi-trap RTN (2 overlapping traps)")
    args = parser.parse_args()
    train(args)
