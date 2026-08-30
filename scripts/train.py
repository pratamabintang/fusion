"""Training CLI script for multispectral object detection."""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from fusion.datasets.llvip import LLVIPDataset, collate_fn
from fusion.engine.evaluator import Evaluator
from fusion.engine.trainer import Trainer
from fusion.models.detector import MS2FusionDetector
from fusion.utils.config import load_config, save_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train MS2Fusion Multispectral Object Detector")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--fusion-type", type=str, choices=["ms2fusion", "ms_ssm", "ic_ssm", "combined"], help="Fusion architecture type")
    parser.add_argument("--data-dir", type=str, help="Path to LLVIP dataset directory")
    parser.add_argument("--img-size", type=int, nargs="+", default=None, help="Image size (H, W)")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--epochs", type=int, help="Total training epochs")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--device", type=str, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable automatic mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--save-dir", type=str, default="runs/train", help="Directory to save experiment outputs")
    parser.add_argument("--weights", type=str, default=None, help="Path to initial weights / pretrained checkpoint .pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--base-channels", type=int, help="Base channel dimension for backbone")
    parser.add_argument("--base-depth", type=int, help="Base depth multiplier for backbone")
    parser.add_argument("--ssm-ratio", type=float, help="SSM channel expansion ratio")
    parser.add_argument("--d-state", type=int, help="SSM state dimension")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader num_workers")
    return parser.parse_args()


def train(cfg: dict) -> dict:
    # Resolve device
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    save_dir = Path(cfg.get("save_dir", "runs/train")) / cfg.get("name", "exp")
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Save effective configuration
    save_config(cfg, str(save_dir / "config.yaml"))
    
    # Image size
    img_size = cfg.get("img_size", [640, 640])
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    else:
        img_size = tuple(img_size)
        
    data_dir = cfg.get("data_dir", "data/LLVIP")
    batch_size = cfg.get("batch_size", 8)
    default_workers = 0 if ("pytest" in sys.modules or os.name == "nt") else min(os.cpu_count() or 4, 8)
    num_workers = cfg.get("workers", default_workers)
    if num_workers is None:
        num_workers = default_workers
    is_cuda = device.startswith("cuda")
    
    train_dataset = LLVIPDataset(data_dir=data_dir, split="train", img_size=img_size)
    val_dataset = LLVIPDataset(data_dir=data_dir, split="test", img_size=img_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True if len(train_dataset) > batch_size else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Build model
    model = MS2FusionDetector(
        fusion_type=cfg.get("fusion_type", "ms2fusion"),
        nc=1,
        base_channels=cfg.get("base_channels", 64),
        base_depth=cfg.get("base_depth", 3),
        ssm_ratio=cfg.get("ssm_ratio", 2.0),
        d_state=cfg.get("d_state", 4),
    ).to(device)
    
    # Load initial weights or resume checkpoint if specified
    weights_path = cfg.get("weights") or cfg.get("resume")
    if weights_path and Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if isinstance(state_dict, dict):
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"[+] Successfully loaded {len(pretrained_dict)}/{len(model_dict)} matching layers from {weights_path}")
            
    print(f"[+] Dataset: {len(train_dataset)} train samples ({len(train_loader)} batches), {len(val_dataset)} val samples ({len(val_loader)} batches)")
    print(f"[+] Device: {device} | Mixed Precision (AMP): {cfg.get('amp', True)} | Batch Size: {batch_size}")
    
    evaluator = Evaluator(model, val_loader, device=device)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.get("lr", 0.001),
        epochs=cfg.get("epochs", 50),
        device=device,
        amp=cfg.get("amp", True),
    )
    
    best_map = -1.0
    history = []
    total_epochs = cfg.get("epochs", 50)
    
    results_csv_path = save_dir / "results.csv"
    with open(results_csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_mAP_0.5,val_mAP_0.5_0.95,precision,recall\n")
        
    for epoch in range(1, total_epochs + 1):
        loss_dict = trainer.train_one_epoch(epoch=epoch, total_epochs=total_epochs)
        train_loss = loss_dict.get("total_loss", 0.0)
        
        val_metrics = evaluator()
        map50 = val_metrics.get("mAP_0.5", 0.0)
        map95 = val_metrics.get("mAP_0.5_0.95", 0.0)
        prec = val_metrics.get("precision", 0.0)
        rec = val_metrics.get("recall", 0.0)
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        })
        
        with open(results_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{map50:.4f},{map95:.4f},{prec:.4f},{rec:.4f}\n")
            
        is_best = map50 > best_map
        if is_best:
            best_map = map50
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": val_metrics,
                "config": cfg,
            }, str(weights_dir / "best.pt"))
            
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": val_metrics,
            "config": cfg,
        }, str(weights_dir / "last.pt"))

        print(f"[Epoch {epoch:2d}/{total_epochs}] Train Loss: {train_loss:.4f} | Val mAP@0.5: {map50*100:.2f}% | Val mAP@0.5:0.95: {map95*100:.2f}%{' -> (Saved Best Checkpoint)' if is_best else ''}")
    
    # Final evaluation
    final_metrics = val_metrics if len(history) > 0 else evaluator()
    
    summary = {
        "final_metrics": final_metrics,
        "best_mAP_0.5": best_map,
        "epochs": len(history),
        "history": history,
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    config = load_config(args.config, overrides)
    print(f"[*] Starting Training: {config.get('fusion_type')} on {config.get('device', 'cuda')}")
    summary = train(config)
    print("[+] Training Complete!")
    print(f"    mAP@0.5: {summary['final_metrics'].get('mAP_0.5', 0.0):.4f}")
    print(f"    mAP@0.5:0.95: {summary['final_metrics'].get('mAP_0.5_0.95', 0.0):.4f}")


if __name__ == "__main__":
    main()
