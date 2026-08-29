"""Multispectral Detection Visual Demo CLI Script."""

import argparse
import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from fusion.datasets.llvip import LLVIPDataset
from fusion.engine.evaluator import non_max_suppression
from fusion.models.detector import MS2FusionDetector
from fusion.utils.visualization import create_side_by_side_vis


def parse_args():
    parser = argparse.ArgumentParser(description="Multispectral Object Detection Visual Demo")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights .pt")
    parser.add_argument("--visible", type=str, default=None, help="Path to single visible image")
    parser.add_argument("--thermal", type=str, default=None, help="Path to single thermal image")
    parser.add_argument("--data-dir", type=str, default="D:/fusion/LLVIP", help="Path to LLVIP dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize from dataset")
    parser.add_argument("--fusion-type", type=str, default="ms2fusion", choices=["ms2fusion", "ms_ssm", "ic_ssm", "combined"])
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, nargs="+", default=[640, 640], help="Image size (H, W)")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--save-dir", type=str, default="runs/demo/exp", help="Directory to save output visualizations")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--base-depth", type=int, default=3)
    return parser.parse_args()


def run_demo(args) -> list[str]:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    img_size = tuple(args.img_size) if isinstance(args.img_size, list) else (args.img_size, args.img_size)
    
    fusion_type = args.fusion_type
    base_channels = args.base_channels
    base_depth = args.base_depth
    
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
    model.eval()
    
    if args.weights and Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
            
    # Gather image pairs
    image_pairs = []
    if args.visible and args.thermal:
        image_pairs.append((Path(args.visible), Path(args.thermal), "custom_pair"))
    else:
        dataset = LLVIPDataset(data_dir=args.data_dir, split=args.split, img_size=img_size)
        num_samples = min(args.num_samples, len(dataset.image_files))
        for vis_p in dataset.image_files[:num_samples]:
            filename = os.path.basename(vis_p)
            img_id = os.path.splitext(filename)[0]
            therm_p = os.path.join(dataset.root, "infrared", dataset.split, filename)
            image_pairs.append((vis_p, therm_p, img_id))
            
    saved_paths = []
    with torch.no_grad():
        for vis_path, therm_path, img_id in image_pairs:
            vis_pil = Image.open(vis_path).convert("RGB")
            therm_pil = Image.open(therm_path).convert("RGB")
            
            # Prepare tensors
            vis_t = TF.to_tensor(TF.resize(vis_pil, img_size)).unsqueeze(0).to(device)
            therm_t = TF.to_tensor(TF.resize(therm_pil, img_size)).unsqueeze(0).to(device)
            
            # Forward pass
            preds, _ = model(vis_t, therm_t)
            
            # NMS
            detections = non_max_suppression(
                preds,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
            )[0]
            
            # Scale coordinates back to original or resized size
            vis_resized = vis_pil.resize((img_size[1], img_size[0]))
            therm_resized = therm_pil.resize((img_size[1], img_size[0]))
            
            comp_img = create_side_by_side_vis(
                vis_img=vis_resized,
                therm_img=therm_resized,
                detections=detections,
                title=f"MS2Fusion ({fusion_type}) - {img_id}",
            )
            
            out_path = save_dir / f"{img_id}_vis.jpg"
            comp_img.save(out_path)
            saved_paths.append(str(out_path))
            print(f"[+] Saved visualization: {out_path} ({len(detections)} detections)")
            
    return saved_paths


def main():
    args = parse_args()
    print(f"[*] Running Multispectral Visual Demo: {args.fusion_type}")
    saved = run_demo(args)
    print(f"[+] Processed and saved {len(saved)} visual outputs to {args.save_dir}")


if __name__ == "__main__":
    main()
