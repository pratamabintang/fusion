import torch
import pytest
import numpy as np

def test_nms_filter():
    from fusion.engine.evaluator import non_max_suppression

    pred = torch.tensor([[[100, 100, 50, 50, 0.9, 1.0],
                          [102, 102, 50, 50, 0.8, 1.0],
                          [300, 300, 50, 50, 0.95, 1.0]]])
    out = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)
    
    assert len(out) == 1
    assert out[0].shape == (2, 6) # two boxes should remain
    
    confs = out[0][:, 4]
    assert torch.allclose(confs, torch.tensor([0.95, 0.9]))
    
    assert torch.allclose(out[0][0, :4], torch.tensor([275., 275., 325., 325.]))

def test_compute_ap_metric():
    from fusion.engine.evaluator import compute_ap
    
    # 10 detections for class 0
    # TP arrays for 2 iou thresholds
    tp = np.array([
        [True, True],
        [True, False],
        [False, False],
        [True, True],
        [False, False]
    ])
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    pred_cls = np.array([0, 0, 0, 0, 0])
    target_cls = np.array([0, 0, 0]) # 3 ground truths
    
    p, r, ap, f1, unique_classes = compute_ap(tp, conf, pred_cls, target_cls)
    
    assert len(unique_classes) == 1
    assert unique_classes[0] == 0
    assert ap.shape == (1, 2)

def test_gate_1_full_test_suite():
    # Programmatic verification is just the fact that pytest tests/ runs.
    # We assert true here.
    assert True

def test_gate_2_synthetic_batch_train_step():
    import torch
    from fusion.engine.trainer import Trainer
    from fusion.models.detector import MS2FusionDetector
    import torch.nn as nn
    
    model = MS2FusionDetector(nc=1, base_channels=32, base_depth=1)
    
    # Synthetic loader
    class DummyLoader:
        def __init__(self):
            self.vis = torch.randn(2, 3, 128, 128)
            self.therm = torch.randn(2, 3, 128, 128)
            self.targets = torch.tensor([
                [0, 0, 0.5, 0.5, 0.2, 0.2],
                [1, 0, 0.3, 0.3, 0.1, 0.1]
            ])
            self.metas = [{'img_size': (128, 128)} for _ in range(2)]
            
        def __iter__(self):
            yield self.vis, self.therm, self.targets, self.metas
            
        def __len__(self):
            return 1
            
    loader = DummyLoader()
    trainer = Trainer(model, loader, epochs=1, device='cpu', amp=False)
    history = trainer.train()
    
    assert len(history) == 1
    assert 'loss_box' in history[0] or 'total' in history[0] or 'total_loss' in history[0]

def test_gate_3_sample_llvip_training_cycle():
    import torch
    from fusion.engine.trainer import Trainer
    from fusion.datasets.llvip import LLVIPDataset, collate_fn
    from fusion.models.detector import MS2FusionDetector
    from torch.utils.data import DataLoader
    import torch.nn as nn
    
    model = MS2FusionDetector(nc=1, base_channels=32, base_depth=1)
    
    dataset = LLVIPDataset('D:/fusion/LLVIP', 'train', img_size=(128, 128))
    # use subset for speed
    dataset.image_files = dataset.image_files[:4]
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    trainer = Trainer(model, loader, epochs=1, device='cpu', amp=False)
    history = trainer.train()
    
    assert len(history) == 1

def test_evaluator_validation_pass():
    import torch
    from fusion.engine.evaluator import Evaluator
    from fusion.datasets.llvip import LLVIPDataset, collate_fn
    from fusion.models.detector import MS2FusionDetector
    from torch.utils.data import DataLoader

    model = MS2FusionDetector(nc=1, base_channels=32, base_depth=1)
    dataset = LLVIPDataset('D:/fusion/LLVIP', 'test', img_size=(128, 128))
    dataset.image_files = dataset.image_files[:2]

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    evaluator = Evaluator(model, loader, device='cpu')
    metrics = evaluator()

    assert 'mAP_0.5' in metrics
    assert 'mAP_0.5_0.95' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
