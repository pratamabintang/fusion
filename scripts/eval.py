"""Evaluation CLI script for multispectral object detection."""

import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from fusion.datasets.llvip import LLVIPDataset, collate_fn
from fusion.engine.evaluator import Evaluator
from fusion.models.detector import MS2FusionDetector
from fusion.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MS2Fusion Detector")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--fusion-type", type=str, default="ms2fusion", choices=["ms2fusion", "ms_ssm", "ic_ssm", "combined"])
    parser.add_argument("--data-dir", type=str, default="data/LLVIP", help="Path to LLVIP dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split ('test' or 'train')")
    parser.add_argument("--img-size", type=int, nargs="+", default=[640, 640], help="Image size (H, W)")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold for NMS")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--base-channels", type=int, default=64, help="Base backbone channels")
    parser.add_argument("--base-depth", type=int, default=3, help="Base backbone depth")
    return parser.parse_args()


def evaluate(args) -> dict:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    img_size = tuple(args.img_size) if isinstance(args.img_size, list) else (args.img_size, args.img_size)
    
    val_dataset = LLVIPDataset(data_dir=args.data_dir, split=args.split, img_size=img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    fusion_type = args.fusion_type
    base_channels = args.base_channels
    base_depth = args.base_depth
    
    # If weights provided, check if config inside
    if args.weights and Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = ckpt["config"]
            fusion_type = cfg.get("fusion_type", fusion_type)
            base_channels = cfg.get("base_channels", base_channels)
            base_depth = cfg.get("base_depth", base_depth)
            
    model = MS2FusionDetector(
        fusion_type=fusion_type,
        nc=1,
        base_channels=base_channels,
        base_depth=base_depth,
    ).to(device)
    
    if args.weights and Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
            
    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=device,
    )
    
    t0 = time.time()
    metrics = evaluator()
    eval_time = (time.time() - t0) * 1000 / max(len(val_dataset), 1)
    
    metrics["latency_ms_per_image"] = eval_time
    metrics["num_images"] = len(val_dataset)
    metrics["fusion_type"] = fusion_type
    return metrics


def main():
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
                
    print(f"[*] Starting Evaluation on LLVIP ({args.split}): {args.fusion_type}")
    metrics = evaluate(args)
    print("=" * 70)
    print(f"{'Class':<12} {'Images':<8} {'P':<8} {'R':<8} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'Latency':<10}")
    print("-" * 70)
    print(f"{'person':<12} {metrics['num_images']:<8} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['mAP_0.5']:<10.4f} {metrics['mAP_0.5_0.95']:<12.4f} {metrics['latency_ms_per_image']:<8.2f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
