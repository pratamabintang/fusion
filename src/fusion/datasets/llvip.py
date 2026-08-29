import os
import glob
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

def parse_voc_xml(xml_path):
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

def xyxy_to_xywh(boxes):
    xywh = torch.empty_like(boxes)
    xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh

def xywh_to_xyxy(boxes):
    xyxy = torch.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy

class LLVIPDataset(Dataset):
    def __init__(self, root, split, img_size=(640, 640), transform=None):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        vis_dir = os.path.join(root, 'visible', split)
        self.image_files = sorted(glob.glob(os.path.join(vis_dir, '*.jpg')))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        vis_path = self.image_files[idx]
        filename = os.path.basename(vis_path)
        ir_path = os.path.join(self.root, 'infrared', self.split, filename)
        xml_path = os.path.join(self.root, 'Annotations', filename.replace('.jpg', '.xml'))
        
        vis_img = Image.open(vis_path).convert('RGB')
        ir_img = Image.open(ir_path).convert('RGB')
        orig_w, orig_h = vis_img.size
        
        vis_img = vis_img.resize(self.img_size, Image.BILINEAR)
        ir_img = ir_img.resize(self.img_size, Image.BILINEAR)
        
        vis_tensor = TF.to_tensor(vis_img)
        ir_tensor = TF.to_tensor(ir_img)
        
        objects = parse_voc_xml(xml_path)
        
        boxes = []
        labels = []
        
        scale_x = self.img_size[0] / orig_w
        scale_y = self.img_size[1] / orig_h
        
        for obj in objects:
            xmin, ymin, xmax, ymax = obj['bbox']
            boxes.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])
            # Assuming 'person' is class 1, or just 0, but standard typically uses integers. Let's use 0 for person.
            labels.append(0)
            
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
            
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }
        
        if self.transform:
            # Not fully implementing complex transforms, but support the kwarg
            pass
            
        return vis_tensor, ir_tensor, target
