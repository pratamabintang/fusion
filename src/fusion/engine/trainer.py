import torch
from fusion.engine.evaluator import Evaluator

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, lr=0.001, epochs=1, device='cpu', amp=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = str(device)
        self.epochs = epochs
        self.amp = amp
        
        if optimizer is None:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=0.0005)
        else:
            self.optimizer = optimizer
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # bfloat16 has full fp32 dynamic range — no GradScaler needed.
        # Only use GradScaler for float16 AMP.
        self.scaler = None
            
        if self.val_loader is not None:
            self.evaluator = Evaluator(model, val_loader, device=device)
        else:
            self.evaluator = None

    def train_one_epoch(self, epoch: int = 1, total_epochs: int = None):
        self.model.train()
        loss_dict = {}
        total_epochs = total_epochs or self.epochs
        
        device_type = 'cuda' if torch.cuda.is_available() and self.device.startswith('cuda') else 'cpu'
        
        try:
            from tqdm import tqdm
            pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch:2d}/{total_epochs}",
                ncols=110,
                leave=True,
            )
        except ImportError:
            pbar = enumerate(self.train_loader)

        for batch_idx, batch in pbar:
            feat_v, feat_t, targets, metas = batch
            feat_v = feat_v.to(self.device, non_blocking=True)
            feat_t = feat_t.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.amp:
                with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss, losses = self.model(feat_v, feat_t, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            else:
                loss, losses = self.model(feat_v, feat_t, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                
            for k, v in losses.items():
                loss_dict[k] = loss_dict.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else float(v))
                
            if hasattr(pbar, "set_postfix") and batch_idx % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "box": f"{losses.get('loss_box', 0.0):.3f}",
                    "obj": f"{losses.get('loss_obj', 0.0):.3f}",
                    "lr": f"{lr:.5f}",
                })
                
        # Average losses
        for k in loss_dict.keys():
            loss_dict[k] /= max(len(self.train_loader), 1)
            
        return loss_dict

    def train(self):
        history = []
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            self.scheduler.step()
            
            metrics = {}
            if self.evaluator is not None:
                metrics = self.evaluator()
                
            summary = {**train_loss, **metrics}
            history.append(summary)
            
        return history
