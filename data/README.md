# Datasets Directory (`data/`)

This directory is the central location for placing multispectral object detection datasets in the Fusion repository.

---

## 📁 Standard Directory Structure

```
data/
├── LLVIP/                      # Default multispectral pedestrian dataset
│   ├── visible/                # Optical RGB images
│   │   ├── train/              # e.g. 010001.jpg, 010002.jpg ...
│   │   └── test/               # e.g. 190001.jpg, 190002.jpg ...
│   ├── infrared/               # Thermal / Infrared images (matching filenames)
│   │   ├── train/              # e.g. 010001.jpg, 010002.jpg ...
│   │   └── test/               # e.g. 190001.jpg, 190002.jpg ...
│   └── Annotations/            # Pascal VOC XML annotation files
│       ├── 010001.xml
│       └── ...
│
├── FLIR/                       # (Optional) FLIR ADAS Thermal Dataset
│   ├── visible/
│   ├── infrared/
│   └── Annotations/
│
├── KAIST/                      # (Optional) KAIST Multispectral Pedestrian Dataset
│   ├── visible/
│   ├── infrared/
│   └── Annotations/
│
├── M3FD/                       # (Optional) M3FD Multispectral Dataset
│   ├── visible/
│   ├── infrared/
│   └── Annotations/
│
└── README.md                   # This guide
```

---

## 📋 Adding a New Experiment Dataset

To add a new dataset for experimentation:

### 1. Match File Names Across Modalities
Ensure each sample has the exact same base filename across all subdirectories:
- **Visible Spectrum**: `data/<DATASET_NAME>/visible/<split>/<image_id>.jpg`
- **Thermal Spectrum**: `data/<DATASET_NAME>/infrared/<split>/<image_id>.jpg`
- **Annotations**: `data/<DATASET_NAME>/Annotations/<image_id>.xml`

### 2. Annotation Format
Annotations follow the standard Pascal VOC XML structure:
```xml
<annotation>
  <filename>010001.jpg</filename>
  <size>
    <width>1280</width>
    <height>1024</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>450</xmin>
      <ymin>320</ymin>
      <xmax>510</xmax>
      <ymax>480</ymax>
    </bndbox>
  </object>
</annotation>
```

### 3. Point Configs or CLI to Your Dataset
You can run any training, evaluation, demo, or benchmark script pointing directly to your dataset:

```powershell
# Train on a new dataset
python scripts/train.py --data-dir data/FLIR --name flir_ms2fusion

# Evaluate on a new dataset
python scripts/eval.py --data-dir data/FLIR --weights runs/train/flir_ms2fusion/weights/best.pt

# Run visual demo on a new dataset
python scripts/demo.py --data-dir data/FLIR --num-samples 10

# Run multi-architecture ablation benchmark
python scripts/benchmark.py --data-dir data/FLIR --output runs/benchmark/flir_ablation.md
```

---

## 💾 Storage & Version Control Policy

- Sample development subsets are tracked in git for continuous integration and automated testing.
- Full datasets (e.g. 15,488 pairs for full LLVIP) should be downloaded and extracted directly into `data/<DATASET_NAME>/`.
- Large image and annotation files can be kept in `data/` and excluded from git if necessary.
