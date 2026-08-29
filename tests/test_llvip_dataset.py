import pytest
from fusion.datasets.llvip import parse_voc_xml, xyxy_to_xywh, xywh_to_xyxy, LLVIPDataset
import xml.etree.ElementTree as ET
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader

def test_parse_voc_xml(tmp_path):
    xml_content = """<?xml version="1.0" ?><annotation>
  <folder>JPEGImages</folder>
  <filename>010001.jpg</filename>
  <size><width>1280</width><height>1024</height><depth>1</depth></size>
  <object>
    <name>person</name>
    <bndbox><xmin>287</xmin><ymin>428</ymin><xmax>351</xmax><ymax>662</ymax></bndbox>
  </object>
  <object>
    <name>person</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>
  </object>
</annotation>"""
    xml_file = tmp_path / "test.xml"
    xml_file.write_text(xml_content)
    
    result = parse_voc_xml(str(xml_file))
    assert len(result) == 2
    assert result[0] == {'class': 'person', 'bbox': [287, 428, 351, 662]}
    assert result[1] == {'class': 'person', 'bbox': [10, 20, 30, 40]}

def test_xyxy_to_xywh():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    expected = torch.tensor([[20.0, 30.0, 20.0, 20.0]])
    assert torch.allclose(xyxy_to_xywh(boxes), expected)

def test_xywh_to_xyxy():
    boxes = torch.tensor([[20.0, 30.0, 20.0, 20.0]])
    expected = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    assert torch.allclose(xywh_to_xyxy(boxes), expected)

def create_mock_llvip(root, num_train=2, num_test=1):
    for split in ['train', 'test']:
        os.makedirs(root / 'visible' / split, exist_ok=True)
        os.makedirs(root / 'infrared' / split, exist_ok=True)
        
    os.makedirs(root / 'Annotations', exist_ok=True)
    
    def create_samples(split, count, start_idx):
        for i in range(count):
            idx_str = f"{start_idx + i:06d}"
            
            # create images
            Image.new('RGB', (64, 64), color='red').save(root / 'visible' / split / f"{idx_str}.jpg")
            Image.new('RGB', (64, 64), color='blue').save(root / 'infrared' / split / f"{idx_str}.jpg")
            
            # create annotation
            xml_content = f"""<?xml version="1.0" ?><annotation>
  <folder>JPEGImages</folder>
  <filename>{idx_str}.jpg</filename>
  <size><width>64</width><height>64</height><depth>1</depth></size>
  <object>
    <name>person</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox>
  </object>
</annotation>"""
            (root / 'Annotations' / f"{idx_str}.xml").write_text(xml_content)

    create_samples('train', num_train, 10000)
    create_samples('test', num_test, 20000)
    
    # Create one empty annotation sample
    idx_str = "020001"
    Image.new('RGB', (64, 64), color='red').save(root / 'visible' / 'test' / f"{idx_str}.jpg")
    Image.new('RGB', (64, 64), color='blue').save(root / 'infrared' / 'test' / f"{idx_str}.jpg")
    xml_content = f"""<?xml version="1.0" ?><annotation>
  <folder>JPEGImages</folder>
  <filename>{idx_str}.jpg</filename>
  <size><width>64</width><height>64</height><depth>1</depth></size>
</annotation>"""
    (root / 'Annotations' / f"{idx_str}.xml").write_text(xml_content)
    
    return root

@pytest.fixture
def mock_llvip(tmp_path):
    return create_mock_llvip(tmp_path)

def test_mock_fixture_creation(mock_llvip):
    assert (mock_llvip / 'visible' / 'train' / '010000.jpg').exists()
    assert (mock_llvip / 'infrared' / 'train' / '010000.jpg').exists()
    assert (mock_llvip / 'Annotations' / '010000.xml').exists()
    
    assert (mock_llvip / 'visible' / 'test' / '020000.jpg').exists()

def test_dataset_len(mock_llvip):
    dataset = LLVIPDataset(mock_llvip, 'train')
    assert len(dataset) == 2

    dataset_test = LLVIPDataset(mock_llvip, 'test')
    assert len(dataset_test) == 2

def test_dataset_getitem(mock_llvip):
    dataset = LLVIPDataset(mock_llvip, 'train', img_size=(32, 32))
    vis, ir, target = dataset[0]
    
    assert vis.shape == (3, 32, 32)
    assert ir.shape == (3, 32, 32)
    
    assert 'boxes' in target
    assert 'labels' in target
    
    boxes = target['boxes']
    assert boxes.shape == (1, 4)
    # Original is 10,10,30,30 on 64x64, scaled to 32x32 -> 5,5,15,15
    assert torch.allclose(boxes, torch.tensor([[5.0, 5.0, 15.0, 15.0]]))
    assert target['labels'].shape == (1,)
    
def test_dataset_empty_annotation(mock_llvip):
    dataset = LLVIPDataset(mock_llvip, 'test', img_size=(32, 32))
    # the second test sample has empty annotation
    vis, ir, target = dataset[1]
    
    assert target['boxes'].shape == (0, 4)
    assert target['labels'].shape == (0,)

def collate_fn(batch):
    vis = torch.stack([b[0] for b in batch])
    ir = torch.stack([b[1] for b in batch])
    targets = [b[2] for b in batch]
    return vis, ir, targets

def test_dataloader(mock_llvip):
    dataset = LLVIPDataset(mock_llvip, 'train')
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    vis, ir, targets = next(iter(loader))
    assert vis.shape == (2, 3, 640, 640)
    assert ir.shape == (2, 3, 640, 640)
    assert len(targets) == 2
