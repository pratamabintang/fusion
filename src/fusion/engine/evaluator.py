"""Evaluation engine for object detection on multispectral datasets.

Implements Non-Maximum Suppression (NMS), 101-point COCO-style average precision (AP)
computation, and dataset-level Evaluator for mAP@0.5 and mAP@0.5:0.95.
"""

import numpy as np
import torch
import torchvision

from fusion.datasets.llvip import xywh_to_xyxy


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[torch.Tensor]:
    """Runs Non-Maximum Suppression (NMS) on model predictions.

    Parameters
    ----------
    prediction : Tensor (B, total_anchors, 5 + nc)
        Raw or decoded detection outputs from the Detect head.
    conf_thres : float
        Confidence threshold for candidate filtering.
    iou_thres : float
        IoU threshold for suppression.
    max_det : int
        Maximum number of output detections per image.

    Returns
    -------
    list of Tensor (N, 6)
        Each tensor contains [x1, y1, x2, y2, conf, class_id] for an image.
    """
    bs = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        conf = x[:, 4] * x[:, 5:].max(1)[0]
        mask = conf > conf_thres
        x = x[mask]
        conf = conf[mask]
        if not x.shape[0]:
            continue

        class_conf, class_id = x[:, 5:].max(1, keepdim=True)

        box = torch.zeros_like(x[:, :4])
        box[:, 0] = x[:, 0] - x[:, 2] / 2
        box[:, 1] = x[:, 1] - x[:, 3] / 2
        box[:, 2] = x[:, 0] + x[:, 2] / 2
        box[:, 3] = x[:, 1] + x[:, 3] / 2

        detections = torch.cat((box, conf.unsqueeze(1), class_id.float()), 1)

        c = detections[:, 5:6] * 7680
        boxes = detections[:, :4] + c
        scores = detections[:, 4]

        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = detections[i]

    return output


def compute_ap(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
) -> tuple:
    """Compute average precision metrics (COCO / Pascal format).

    Parameters
    ----------
    tp : np.ndarray (N, n_iou_thresholds)
        Boolean array indicating true positives across IoU thresholds.
    conf : np.ndarray (N,)
        Confidence scores for predictions.
    pred_cls : np.ndarray (N,)
        Predicted class IDs.
    target_cls : np.ndarray (M,)
        Ground truth class IDs.

    Returns
    -------
    tuple
        (p_curve, r_curve, ap, f1_curve, unique_classes)
    """
    if len(tp) == 0:
        return np.zeros(1000), np.zeros(1000), np.zeros((1, 10)), np.zeros(1000), np.array([0])

    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    f1_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()

        if n_p == 0 and n_l == 0:
            continue
        elif n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)

        for j in range(tp.shape[1]):
            mrec = np.concatenate(([0.0], recall[:, j], [1.0]))
            mpre = np.concatenate(([1.0], precision[:, j], [0.0]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            x = np.linspace(0, 1, 101)
            ap[ci, j] = np.mean(np.interp(x, mrec, mpre))

    return p_curve, r_curve, ap, f1_curve, unique_classes


class Evaluator:
    """Evaluation engine for computing object detection metrics on LLVIP.

    Parameters
    ----------
    model : nn.Module
        The MS2FusionDetector model.
    val_loader : DataLoader
        DataLoader yielding validation batches.
    conf_thres : float
        Confidence threshold for candidate detection boxes (default: 0.001).
    iou_thres : float
        NMS IoU threshold (default: 0.6).
    device : str
        Target device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model,
        val_loader,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        device: str = "cpu",
    ):
        self.model = model
        self.val_loader = val_loader
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

    def __call__(self) -> dict[str, float]:
        """Execute full evaluation over the validation set."""
        self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10)
        stats = []

        with torch.no_grad():
            for batch in self.val_loader:
                vis, therm, targets, metas = batch
                vis = vis.to(self.device)
                therm = therm.to(self.device)

                preds, _ = self.model(vis, therm)
                detections = non_max_suppression(
                    preds, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                )

                for i, det in enumerate(detections):
                    gt = targets[targets[:, 0] == i] if targets.shape[0] > 0 else torch.zeros((0, 6))
                    nl = len(gt)
                    tcls = gt[:, 1].tolist() if nl else []

                    if len(det) == 0:
                        if nl:
                            stats.append((
                                np.zeros((0, 10), dtype=bool),
                                np.zeros(0),
                                np.zeros(0),
                                np.array(tcls),
                            ))
                        continue

                    if nl == 0:
                        stats.append((
                            np.zeros((len(det), 10), dtype=bool),
                            det[:, 4].cpu().numpy(),
                            det[:, 5].cpu().numpy(),
                            np.zeros(0),
                        ))
                        continue

                    # Ground truth boxes in image coordinates
                    h, w = metas[i]["img_size"]
                    gt_boxes_norm = gt[:, 2:6]
                    gt_boxes = xywh_to_xyxy(gt_boxes_norm)
                    gt_boxes[:, [0, 2]] *= w
                    gt_boxes[:, [1, 3]] *= h

                    # Match detections with ground truth
                    pred_boxes = det[:, :4].cpu()
                    pred_cls = det[:, 5].cpu()
                    gt_cls = gt[:, 1].cpu()

                    # Compute IoU matrix (N_pred, N_gt)
                    from torchvision.ops import box_iou
                    ious = box_iou(pred_boxes, gt_boxes)

                    correct = np.zeros((len(det), 10), dtype=bool)
                    for j, iou_thresh in enumerate(iouv):
                        # Match greedy by highest IoU
                        matched_gt = set()
                        for p_idx in range(len(det)):
                            best_iou = 0.0
                            best_gt = -1
                            for g_idx in range(nl):
                                if g_idx in matched_gt:
                                    continue
                                if pred_cls[p_idx] != gt_cls[g_idx]:
                                    continue
                                if ious[p_idx, g_idx] >= iou_thresh and ious[p_idx, g_idx] > best_iou:
                                    best_iou = ious[p_idx, g_idx].item()
                                    best_gt = g_idx
                            if best_gt >= 0:
                                matched_gt.add(best_gt)
                                correct[p_idx, j] = True

                    stats.append((
                        correct,
                        det[:, 4].cpu().numpy(),
                        det[:, 5].cpu().numpy(),
                        np.array(tcls),
                    ))

        if len(stats) == 0:
            return {"mAP_0.5": 0.0, "mAP_0.5_0.95": 0.0, "precision": 0.0, "recall": 0.0}

        tp = np.concatenate([x[0] for x in stats], axis=0) if sum(len(x[0]) for x in stats) > 0 else np.zeros((0, 10), dtype=bool)
        conf = np.concatenate([x[1] for x in stats]) if sum(len(x[1]) for x in stats) > 0 else np.zeros(0)
        pred_cls = np.concatenate([x[2] for x in stats]) if sum(len(x[2]) for x in stats) > 0 else np.zeros(0)
        target_cls = np.concatenate([x[3] for x in stats]) if sum(len(x[3]) for x in stats) > 0 else np.zeros(0)

        _, _, ap, _, _ = compute_ap(tp, conf, pred_cls, target_cls)
        mAP50 = ap[:, 0].mean() if len(ap) > 0 else 0.0
        mAP = ap.mean() if len(ap) > 0 else 0.0

        p = tp[:, 0].sum() / max(len(tp), 1) if len(tp) > 0 else 0.0
        r = tp[:, 0].sum() / max(len(target_cls), 1) if len(target_cls) > 0 else 0.0

        return {
            "mAP_0.5": float(mAP50),
            "mAP_0.5_0.95": float(mAP),
            "precision": float(p),
            "recall": float(r),
        }

