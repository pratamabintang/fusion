"""LLVIP Paired Dataset — visible/thermal pedestrian detection pipeline.

Loads paired visible (RGB) and thermal (infrared) images with Pascal VOC XML
annotations from the LLVIP dataset structure.
"""

import os
import glob
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def parse_voc_xml(xml_path: str) -> list[dict]:
    """Parse a Pascal VOC XML annotation file.

    Returns a list of dicts, each with ``'class'`` (str) and
    ``'bbox'`` ([xmin, ymin, xmax, ymax] as ints).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({'class': name, 'bbox': [xmin, ymin, xmax, ymax]})
    return objects


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from corner ``[x1, y1, x2, y2]`` to center
    ``[x_center, y_center, width, height]`` format.

    Parameters
    ----------
    boxes : Tensor (N, 4)

    Returns
    -------
    Tensor (N, 4)
    """
    xywh = torch.empty_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from center ``[x_c, y_c, w, h]`` to corner
    ``[x1, y1, x2, y2]`` format.

    Parameters
    ----------
    boxes : Tensor (N, 4)

    Returns
    -------
    Tensor (N, 4)
    """
    xyxy = torch.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy


def _resolve_dataset_root(root: str | None = None) -> str:
    """Resolve dataset directory path with data/ folder prioritization."""
    if root is not None:
        if os.path.exists(root):
            return root
        if os.path.exists(os.path.join("data", root)):
            return os.path.join("data", root)
        return root
    for candidate in ["data/LLVIP", "D:/fusion/data/LLVIP", "LLVIP", "D:/fusion/LLVIP"]:
        if os.path.exists(candidate):
            return candidate
    return "data/LLVIP"


class LLVIPDataset(Dataset):
    """PyTorch Dataset for the LLVIP paired visible/thermal pedestrian dataset.

    Each sample returns ``(visible_tensor, thermal_tensor, target, image_meta)``
    where:

    - ``visible_tensor``: (3, H, W) float32 normalised [0, 1]
    - ``thermal_tensor``: (3, H, W) float32 normalised [0, 1]
    - ``target``: dict with ``'boxes'`` (N, 4) xyxy and ``'labels'`` (N,) int64
    - ``image_meta``: dict with ``'filename'``, ``'orig_size'`` (H, W),
      ``'img_size'`` (H, W)
    """

    def __init__(self, root=None, split="train", img_size=(640, 640), transform=None, data_dir=None):
        self.root = _resolve_dataset_root(root or data_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform

        vis_dir = os.path.join(self.root, 'visible', split)
        self.image_files = sorted(glob.glob(os.path.join(vis_dir, '*.jpg')))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        vis_path = self.image_files[idx]
        filename = os.path.basename(vis_path)
        thermal_path = os.path.join(
            self.root, 'infrared', self.split, filename,
        )
        xml_path = os.path.join(
            self.root, 'Annotations', filename.replace('.jpg', '.xml'),
        )

        vis_img = Image.open(vis_path).convert('RGB')
        thermal_img = Image.open(thermal_path).convert('RGB')
        orig_w, orig_h = vis_img.size

        vis_img = vis_img.resize(self.img_size, Image.BILINEAR)
        thermal_img = thermal_img.resize(self.img_size, Image.BILINEAR)

        visible_tensor = TF.to_tensor(vis_img)
        thermal_tensor = TF.to_tensor(thermal_img)

        objects = parse_voc_xml(xml_path)

        boxes = []
        labels = []

        scale_x = self.img_size[0] / orig_w
        scale_y = self.img_size[1] / orig_h

        for obj in objects:
            xmin, ymin, xmax, ymax = obj['bbox']
            boxes.append([
                xmin * scale_x, ymin * scale_y,
                xmax * scale_x, ymax * scale_y,
            ])
            labels.append(0)  # single-class: person = 0

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
        }

        image_meta = {
            'filename': filename,
            'orig_size': (orig_h, orig_w),
            'img_size': (self.img_size[1], self.img_size[0]),
        }

        if self.transform:
            visible_tensor, thermal_tensor, target = self.transform(
                visible_tensor, thermal_tensor, target,
            )

        return visible_tensor, thermal_tensor, target, image_meta

def collate_fn(batch):
    """
    Collate function for LLVIP dataset.
    batch is list of tuples: (visible_tensor, thermal_tensor, target, image_meta)
    Returns:
        vis_batch: (B, 3, H, W)
        therm_batch: (B, 3, H, W)
        targets: (N, 6) -> [image_idx, class_id, x_c, y_c, w, h]
        metas: list of image_meta
    """
    vis_batch = []
    therm_batch = []
    targets = []
    metas = []
    
    for i, (v, t, target, meta) in enumerate(batch):
        vis_batch.append(v)
        therm_batch.append(t)
        metas.append(meta)
        
        boxes = target['boxes'] # (n, 4) xyxy
        labels = target['labels'] # (n,)
        
        if boxes.shape[0] > 0:
            # Convert xyxy to xywh
            xywh = xyxy_to_xywh(boxes)
            
            # Normalize to [0, 1] using img_size from meta
            h, w = meta['img_size']
            xywh[:, 0] /= w
            xywh[:, 2] /= w
            xywh[:, 1] /= h
            xywh[:, 3] /= h
            
            # Create (n, 6) tensor
            img_idx = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
            cls_id = labels.unsqueeze(1).float()
            
            tgt = torch.cat((img_idx, cls_id, xywh), dim=1)
            targets.append(tgt)
            
    vis_batch = torch.stack(vis_batch, dim=0)
    therm_batch = torch.stack(therm_batch, dim=0)
    
    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
        
    return vis_batch, therm_batch, targets, metas


