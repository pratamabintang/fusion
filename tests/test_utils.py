import os
import pytest
from PIL import Image
import numpy as np
import torch
import yaml
from pathlib import Path

from fusion.utils import load_config, save_config, draw_detections, create_side_by_side_vis

def test_config_loading_and_override(tmp_path):
    # Test loading specific config
    config_path = tmp_path / "test_config.yaml"
    test_config = {"fusion_type": "test_type", "batch_size": 16}
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
        
    config = load_config(str(config_path))
    assert config["fusion_type"] == "test_type"
    assert config["batch_size"] == 16
    
    # Test override
    overrides = {"batch_size": 32, "lr": 0.01}
    config = load_config(str(config_path), overrides)
    assert config["fusion_type"] == "test_type"
    assert config["batch_size"] == 32
    assert config["lr"] == 0.01

def test_all_preset_configs():
    # Test default fallback and all presets
    config_dir = Path("D:/fusion/configs")
    configs = ["default.yaml", "ms2fusion_baseline.yaml", "ms_ssm.yaml", "ic_ssm.yaml", "combined.yaml"]
    
    for conf in configs:
        conf_path = config_dir / conf
        # Ensure it exists before testing to not fail purely on missing file, 
        # but the test will fail on parse if not exists since we'll write it.
        config = load_config(str(conf_path))
        assert "fusion_type" in config
        assert "batch_size" in config
        
def test_draw_detections():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    detections = torch.tensor([[10, 10, 50, 50, 0.9, 0]]) # x1, y1, x2, y2, conf, cls
    
    res_img = draw_detections(img, detections)
    assert isinstance(res_img, Image.Image)
    assert res_img.size == (100, 100)

def test_side_by_side_visualization():
    vis_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    therm_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    detections = np.array([[10, 10, 50, 50, 0.9, 0]])
    
    res_img = create_side_by_side_vis(vis_img, therm_img, detections)
    assert isinstance(res_img, Image.Image)
    assert res_img.size[0] == 200 # side by side
    assert res_img.size[1] >= 100 # usually a header is added for titles, so height >= 100
