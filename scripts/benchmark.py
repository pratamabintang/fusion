"""Multi-Architecture Comparative Ablation Benchmark Runner CLI."""

import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from fusion.datasets.llvip import LLVIPDataset, collate_fn
from fusion.engine.evaluator import Evaluator
from fusion.engine.trainer import Trainer
from fusion.models.detector import MS2FusionDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Architecture Ablation Benchmark Runner")
    parser.add_argument("--data-dir", type=str, default="D:/fusion/LLVIP", help="Path to LLVIP dataset")
    parser.add_argument("--split", type=str, default="test", help="Evaluation split")
    parser.add_argument("--img-size", type=int, nargs="+", default=[640, 640], help="Image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--quick", action="store_true", help="Run fast benchmark mode with small subsets")
    parser.add_argument("--epochs", type=int, default=0, help="Train each variant for N epochs before evaluating (0 = eval untrained/weights)")
    parser.add_argument("--output", type=str, default="runs/benchmark/ablation_report.md", help="Path to output markdown report")
    parser.add_argument("--base-channels", type=int, default=32, help="Backbone base channels for benchmark")
    parser.add_argument("--base-depth", type=int, default=1, help="Backbone base depth for benchmark")
    return parser.parse_args()


def benchmark(args) -> list[dict]:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    img_size = tuple(args.img_size) if isinstance(args.img_size, list) else (args.img_size, args.img_size)
    if args.quick:
        img_size = (128, 128)
        
    modes = [
        ("ms2fusion", "Baseline Replication (CP + SP + FF-SSM)"),
        ("ms_ssm", "Innovation 1 (MS-SSM Memory Bridge P3->P4->P5)"),
        ("ic_ssm", "Innovation 2 (IC-SSM Illumination Adaptive Modulation)"),
        ("combined", "Combined (MS-SSM Bridge + IC-SSM Modulation)"),
    ]
    
    val_dataset = LLVIPDataset(data_dir=args.data_dir, split=args.split, img_size=img_size)
    if args.quick:
        val_dataset.image_files = val_dataset.image_files[:4]
        
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)), shuffle=False, collate_fn=collate_fn)
    
    if args.epochs > 0:
        train_dataset = LLVIPDataset(data_dir=args.data_dir, split="train", img_size=img_size)
        if args.quick:
            train_dataset.image_files = train_dataset.image_files[:4]
        train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True, collate_fn=collate_fn)
    else:
        train_loader = None
        
    results = []
    
    for mode_key, mode_desc in modes:
        print(f"\n[*] Benchmarking Variant: {mode_key} ({mode_desc})")
        
        # Instantiate model
        model = MS2FusionDetector(
            fusion_type=mode_key,
            nc=1,
            base_channels=args.base_channels,
            base_depth=args.base_depth,
        ).to(device)
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        param_millions = total_params / 1e6
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Training if requested
        if args.epochs > 0 and train_loader is not None:
            print(f"    Training {mode_key} for {args.epochs} epoch(s)...")
            trainer = Trainer(model, train_loader, epochs=args.epochs, device=device, amp=False)
            trainer.train()
            
        # Inference Latency Benchmark (10 warm-up + 10 timed passes)
        model.eval()
        dummy_v = torch.randn(1, 3, *img_size, device=device)
        dummy_t = torch.randn(1, 3, *img_size, device=device)
        
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_v, dummy_t)
                
            t0 = time.perf_counter()
            num_passes = 10
            for _ in range(num_passes):
                _ = model(dummy_v, dummy_t)
            t1 = time.perf_counter()
            
        latency_ms = ((t1 - t0) / num_passes) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        
        # Accuracy evaluation
        evaluator = Evaluator(model, val_loader, device=device)
        metrics = evaluator()
        
        entry = {
            "mode": mode_key,
            "description": mode_desc,
            "params_m": param_millions,
            "size_mb": size_mb,
            "latency_ms": latency_ms,
            "fps": fps,
            "mAP_0.5": metrics.get("mAP_0.5", 0.0),
            "mAP_0.5_0.95": metrics.get("mAP_0.5_0.95", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
        }
        results.append(entry)
        
        print(f"    Params: {param_millions:.2f}M | Size: {size_mb:.2f}MB | Latency: {latency_ms:.2f}ms ({fps:.1f} FPS)")
        print(f"    mAP@0.5: {entry['mAP_0.5']:.4f} | mAP@0.5:0.95: {entry['mAP_0.5_0.95']:.4f}")
        
    # Generate Markdown Report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_content = "# MS2Fusion Multi-Architecture Comparative Ablation Benchmark\n\n"
    md_content += f"- **Dataset**: LLVIP (`{args.split}` split)\n"
    md_content += f"- **Resolution**: {img_size[0]}x{img_size[1]}\n"
    md_content += f"- **Device**: `{device}`\n\n"
    md_content += "## 📊 Comparative Performance Matrix\n\n"
    md_content += "| Architecture / Mode | Description | Params (M) | Size (MB) | Latency (ms) | FPS | mAP@0.5 (%) | mAP@0.5:0.95 (%) |\n"
    md_content += "|---|---|---|---|---|---|---|---|\n"
    
    for r in results:
        md_content += (
            f"| `{r['mode']}` | {r['description']} | {r['params_m']:.2f} | "
            f"{r['size_mb']:.2f} | {r['latency_ms']:.2f} | {r['fps']:.1f} | "
            f"{r['mAP_0.5'] * 100:.2f}% | {r['mAP_0.5_0.95'] * 100:.2f}% |\n"
        )
        
    md_content += "\n## 💡 Key Architectural Insights\n\n"
    md_content += "1. **Baseline Replication (`ms2fusion`)**: Integrates CP-SSM projection exchange and bidirectional FF-SSM with SE gating across independent scales.\n"
    md_content += "2. **Cross-Scale State Memory (`ms_ssm`)**: Propagates persistent hidden memory vectors across $P_3 \\rightarrow P_4 \\rightarrow P_5$ with negligible parameter increase.\n"
    md_content += r"3. **Illumination Modulation (`ic_ssm`)**: Dynamically weights continuous-to-discrete step sizes ($\Delta$) and transition matrices ($A$) according to optical luminance conditions." + "\n"
    md_content += "4. **Combined (`combined`)**: Unified architecture integrating memory persistence and illumination awareness.\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\n[+] Ablation benchmark report written to: {output_path}")
    print("\n" + md_content)
    return results


def main():
    args = parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
