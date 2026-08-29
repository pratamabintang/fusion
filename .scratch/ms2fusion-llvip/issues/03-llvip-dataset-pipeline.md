# 03: LLVIP Paired Dataset Loader & Pascal VOC XML Annotation Pipeline

**What to build:** A robust PyTorch Dataset and DataLoader pipeline that loads aligned visible and infrared image pairs with their corresponding Pascal VOC XML annotations, alongside synthetic mock fixtures for fast deterministic TDD testing.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Implement XML annotation parser for Pascal VOC format extracting bounding boxes and class labels (`person`).
- [x] Implement `LLVIPDataset` supporting synchronized loading of `visible/` (RGB) and `infrared/` (grayscale/3-channel) image pairs for train and test splits.
- [x] Implement synchronized data augmentations (resizing to 640×640 / 640×512, normalization, horizontal flip, mosaic augmentation).
- [x] Provide synthetic paired-image dataset fixtures for fast, deterministic unit and integration tests.
- [x] Unit tests verify dataset indexing, bounding box format transformations ($[x_1, y_1, x_2, y_2] \leftrightarrow [x_c, y_c, w, h]$), and DataLoader collation.
