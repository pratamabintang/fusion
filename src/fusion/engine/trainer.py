import torch
from torch.cuda.amp import autocast, GradScaler
from fusion.engine.evaluator import Evaluator

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, optimizer=None, lr=0.001, epochs=1, device='cpu', amp=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.amp = amp
        
        if optimizer is None:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=0.0005)
        else:
            self.optimizer = optimizer
            
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        if self.amp:
            # Check if CUDA is available for amp scaler
            self.scaler = GradScaler(enabled=torch.cuda.is_available())
        else:
            self.scaler = None
            
        if self.val_loader is not None:
            self.evaluator = Evaluator(model, val_loader, device=device)
        else:
            self.evaluator = None

    def train_one_epoch(self):
        self.model.train()
        loss_dict = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            vis, therm, targets, metas = batch
            vis = vis.to(self.device)
            therm = therm.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.amp:
                # Use amp package properly
                with autocast(enabled=torch.cuda.is_available()):
                    loss, losses = self.model(vis, therm, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, losses = self.model(vis, therm, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
                
            for k, v in losses.items():
                loss_dict[k] = loss_dict.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else float(v))
                
        # Average losses
        for k in loss_dict.keys():
            loss_dict[k] /= len(self.train_loader)
            
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
