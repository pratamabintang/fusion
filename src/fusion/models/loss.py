import torch
import torch.nn as nn
import math

def bbox_iou(box1, box2, xywh=True, CIoU=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    if xywh:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)
    return iou

class YOLOLoss:
    def __init__(self, model, nc=1, hyp=None):
        self.nc = nc
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        
        # balances for obj loss: P3, P4, P5
        self.balance = [4.0, 1.0, 0.4]
        self.anchors = model.anchors
        self.nl = model.nl
        self.na = model.na

    def __call__(self, p, targets):
        # p is list of predictions from each layer
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                
                iou = bbox_iou(pbox.T, tbox[i], xywh=True, CIoU=True)
                lbox += (1.0 - iou).mean()
                
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = score_iou
                
                # Classification
                if self.nc > 1:
                    t = torch.zeros_like(ps[:, 5:], device=device)
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.bce_cls(ps[:, 5:], t)

            lobj += self.bce_obj(pi[..., 4], tobj) * self.balance[i]
            
        total_loss = lbox * 0.05 + lobj * 1.0 + lcls * 0.5
        
        loss_dict = {
            'loss_box': lbox.item(),
            'loss_obj': lobj.item(),
            'loss_cls': lcls.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict

    def build_targets(self, p, targets):
        # Targets shape: (nt, 6) -> image_idx, class_id, x, y, w, h
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        
        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g
        
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4.0  # Anchor threshold
                t = t[j]
                
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            b, c = t[:, 0].long(), t[:, 1].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            
            a = t[:, 6].long()
            
            indices.append((b, a, gj.clamp_(0, int(gain[3]) - 1), gi.clamp_(0, int(gain[2]) - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
            
        return tcls, tbox, indices, anch
