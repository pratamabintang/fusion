"""Training CLI script for multispectral object detection."""

import argparse
import json
import os
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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--base-channels", type=int, help="Base channel dimension for backbone")
    parser.add_argument("--base-depth", type=int, help="Base depth multiplier for backbone")
    parser.add_argument("--ssm-ratio", type=float, help="SSM channel expansion ratio")
    parser.add_argument("--d-state", type=int, help="SSM state dimension")
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
        
    data_dir = cfg.get("data_dir", "D:/fusion/LLVIP")
    batch_size = cfg.get("batch_size", 8)
    
    train_dataset = LLVIPDataset(data_dir=data_dir, split="train", img_size=img_size)
    val_dataset = LLVIPDataset(data_dir=data_dir, split="test", img_size=img_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True if len(train_dataset) > batch_size else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
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
    
    # Resume checkpoint if specified
    if cfg.get("resume") and Path(cfg["resume"]).exists():
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
            
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
    
    history = trainer.train()
    
    # Evaluate final
    final_metrics = evaluator()
    
    # Save checkpoints
    torch.save({
        "epoch": cfg.get("epochs", 50),
        "model_state_dict": model.state_dict(),
        "metrics": final_metrics,
        "config": cfg,
    }, str(weights_dir / "last.pt"))
    
    torch.save({
        "epoch": cfg.get("epochs", 50),
        "model_state_dict": model.state_dict(),
        "metrics": final_metrics,
        "config": cfg,
    }, str(weights_dir / "best.pt"))
    
    # Save results summary
    summary = {
        "final_metrics": final_metrics,
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
