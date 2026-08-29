# 04: Two-Stream YOLOv5 Detector Architecture with MS2Fusion Seams

**What to build:** The complete end-to-end multispectral object detector architecture incorporating two parallel CSPDarkNet53 backbones, multi-scale MS2Fusion blocks at $P_3, P_4, P_5$ pyramid levels, a PANet neck, and anchor-based detection heads with composite loss computation.

**Blocked by:** 02 (MS2Fusion Parametric Interaction Modules)

**Status:** ready-for-agent

- [ ] Implement dual-stream CSPDarkNet53 backbone with independent feature extraction for visible and thermal modalities.
- [ ] Connect multi-scale features ($P_3, P_4, P_5$) to `MS2FusionBlock` instances at each scale.
- [ ] Implement PANet neck aggregating multi-scale fused features with top-down and bottom-up pathways.
- [ ] Implement anchor-based detection head with bounding box regression, objectness, and class probability prediction.
- [ ] Implement composite YOLO loss function (GIoU/CIoU box loss, BCE objectness loss, BCE classification loss).
- [ ] Integration test verifies an end-to-end forward pass and backward step on CPU with synthetic batch without NaN/Inf.
