# LLVIP Dataset

This directory contains the local LLVIP dataset copy used by the repository.

## Expected layout

```
LLVIP/
  visible/
    train/
    test/
  infrared/
    train/
    test/
  Annotations/
```

## Description

- `visible/`: RGB images for the visible spectrum.
- `infrared/`: grayscale infrared images aligned with the visible images.
- `Annotations/`: XML annotations in PASCAL VOC-style format.

## Usage

The dataset loader used by `baseline/train.py`, `baseline/detect.py`, and `baseline/val.py` expects matching file names across the visible, infrared, and annotation folders.

## Notes

- The dataset is currently stored locally for development.
- If the dataset is large, keep the source data outside version control and update the repository paths accordingly.
