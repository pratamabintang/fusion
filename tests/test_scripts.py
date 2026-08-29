import os
from pathlib import Path
import pytest
import torch

from scripts.train import train
from scripts.eval import evaluate
from scripts.demo import run_demo
from scripts.benchmark import benchmark


def test_train_cli_execution(tmp_path):
    save_dir = tmp_path / "runs" / "train"
    cfg = {
        "fusion_type": "ms2fusion",
        "data_dir": "D:/fusion/LLVIP",
        "img_size": [128, 128],
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.001,
        "device": "cpu",
        "amp": False,
        "name": "test_train_run",
        "save_dir": str(save_dir),
        "base_channels": 16,
        "base_depth": 1,
    }
    
    summary = train(cfg)
    
    run_dir = save_dir / "test_train_run"
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config.yaml").exists()
    assert "final_metrics" in summary


def test_eval_cli_execution():
    class Args:
        weights = None
        config = None
        fusion_type = "ms_ssm"
        data_dir = "D:/fusion/LLVIP"
        split = "test"
        img_size = [128, 128]
        batch_size = 2
        device = "cpu"
        conf_thres = 0.001
        iou_thres = 0.6
        base_channels = 16
        base_depth = 1
        
    metrics = evaluate(Args())
    assert "mAP_0.5" in metrics
    assert "mAP_0.5_0.95" in metrics
    assert "latency_ms_per_image" in metrics
    assert metrics["fusion_type"] == "ms_ssm"


def test_demo_cli_execution(tmp_path):
    demo_save_dir = tmp_path / "runs" / "demo" / "test_exp"
    
    class Args:
        weights = None
        visible = None
        thermal = None
        data_dir = "D:/fusion/LLVIP"
        split = "test"
        num_samples = 2
        fusion_type = "combined"
        conf_thres = 0.25
        iou_thres = 0.45
        img_size = [128, 128]
        device = "cpu"
        save_dir = str(demo_save_dir)
        base_channels = 16
        base_depth = 1
        
    saved = run_demo(Args())
    assert len(saved) == 2
    for p in saved:
        assert Path(p).exists()
        assert Path(p).suffix.lower() in [".jpg", ".png"]


def test_benchmark_cli_execution(tmp_path):
    report_path = tmp_path / "runs" / "benchmark" / "test_ablation.md"
    
    class Args:
        data_dir = "D:/fusion/LLVIP"
        split = "test"
        img_size = [128, 128]
        batch_size = 2
        device = "cpu"
        quick = True
        epochs = 0
        output = str(report_path)
        base_channels = 16
        base_depth = 1
        
    results = benchmark(Args())
    assert len(results) == 4
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Baseline Model (Pure MS2Fusion)" in content
        assert "MS-SSM (Multi-Scale State-Space Memory Bridge)" in content
        assert "IC-SSM (Illumination & Contrast Adaptive State Modulation)" in content
