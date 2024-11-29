from typing import Any, Dict, List, Tuple
import time
from datetime import datetime
from os import path, makedirs
from typing import Union, Optional
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F 
from torch import autograd
from torch import func

def _build_saving_path(base_name: str, base_folder: str, additional_str_id: str = "") -> str:
    _time = time.time()
    _date = str(datetime.fromtimestamp(_time)).split(" ")[0].replace("-", "_")

    makedirs(base_folder, exist_ok=True)
    filename = f"{base_name}_{_date}_{additional_str_id}"
    
    return str(path.join(base_folder, f"{filename}.pth"))

def save_trained_model(model: torch.nn.Module, train_dataset: str, base_folder: str = "./saved_models/", additional_str_id: str = "") -> str:
    print("Saving model...")
    model_path = _build_saving_path(f"{type(model).__name__}_{train_dataset}", base_folder=base_folder, additional_str_id=additional_str_id)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in {model_path}")
    return model_path

def load_model(model_class: torch.nn.Module, kwargs_dict: Dict[str, Any], path: str, map_location="cuda") -> torch.nn.Module:
    model: torch.nn.Module = model_class(**kwargs_dict)
    state_dict = torch.load(path, map_location=torch.device(map_location))
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

class ModelCheckpoint:
    def __init__(
        self, 
        model:          torch.nn.Module, 
        optimizer:      torch.optim.Optimizer,
        mode:           str, 
        base_folder:    str = "./checkpoints/"
    ) -> None:
    
        if mode not in  {"best", "last"}:
            raise ValueError(f"Unsupported mode {mode}. Use 'best' or 'last'")
        
        self.model:       torch.nn.Module       = model
        self.optimizer:   torch.optim.Optimizer = optimizer
        self.mode:        str                   = mode
        self.base_folder: str                   = base_folder
        self.best_loss:   torch.FloatType       = torch.inf
        self.saved:       bool                  = False

    def load_checkpoint(self) -> Tuple[torch.FloatType, torch.FloatType, int]:
        if not self.saved:
            raise RuntimeError("You are trying to load a checkpoint not yet saved.")
        
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        train_loss = checkpoint["train_loss"]
        val_loss   = checkpoint["val_loss"]
        epoch      = checkpoint["epoch"]

        return (train_loss, val_loss, epoch)        


    def __save_checkpoint(self, epoch: int, train_loss: torch.FloatType, val_loss: torch.FloatType, additional_str_id: str = "") -> None:
        self.checkpoint_path: str = _build_saving_path(type(self.model).__name__, self.base_folder, additional_str_id)
        torch.save({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, self.checkpoint_path)
        self.saved = True
        

    def __call__(self, epoch: int, train_loss: torch.FloatType, val_loss: torch.FloatType, additional_str_id: str = "") -> None:
        if self.mode == "last":
            self.__save_checkpoint(epoch, train_loss, val_loss, additional_str_id)
        elif self.mode == "best":
            if self.best_loss - val_loss > 1e-5:
                self.__save_checkpoint(epoch, train_loss, val_loss, additional_str_id)
                self.best_loss = val_loss


class LRDecayWithPatience:
    def __init__(
        self, 
        optimizer:      torch.optim.Optimizer, 
        patience:       int = 1, 
        min_lr:         torch.FloatType = 1e-7,
        threshold:      torch.FloatType = 1e-4, 
        decay_factor:   torch.FloatType = 0.9,
        verbose:        bool            = True
    ) -> None:
        
        self.optimizer:     torch.optim.Optimizer = optimizer
        self.patience:      int                   = patience
        self.min_lr:        torch.FloatType       = min_lr
        self.decay_factor:  torch.FloatType       = decay_factor
        self.threshold:     torch.FloatType       = threshold

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = self.optimizer,
            mode = "min",
            patience = self.patience,
            min_lr = self.min_lr,
            factor = self.decay_factor,
            threshold=self.threshold,
            verbose=verbose
        )

    def __call__(self, val_loss: torch.FloatType) -> None:
        self.lr_scheduler.step(val_loss)

    def get_lr(self) -> torch.FloatType:
        return self.optimizer.param_groups[0]["lr"]
    
class LRMultiStep:
    def __init__(
        self, 
        optimizer:      torch.optim.Optimizer, 
        milestones:     List[float] = [],
        gamma:          float       = 0.1,
        last_epoch:     int         = -1,
        verbose:        bool            = True
    ) -> None:
        
        self.optimizer:     torch.optim.Optimizer = optimizer
        self.milestones:    List[int]             = milestones
        self.gamma:         float                 = gamma
        self.last_epoch:    int                   = last_epoch

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer = self.optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
            verbose=verbose
        )

    def __call__(self, epoch: int) -> None:
        self.lr_scheduler.step(epoch)

    def get_lr(self) -> torch.FloatType:
        return self.optimizer.param_groups[0]["lr"]

class EarlyStopping:
    def __init__(
        self, patience: int = 10, 
        epsilon: torch.FloatType = 1e-4,
        warmup: int = 0,
        verbose: bool = True
    ) -> None:
    
        self.patience:  int             = patience
        self.epsilon:   float           = epsilon
        self.counter:   int             = 0
        self.best_loss: torch.FloatType = torch.inf
        self.warmup:    int             = warmup
        self.verbose:   bool            = verbose


    def __call__(self, val_loss: int, current_epoch: int) -> bool:
        if current_epoch + 1 < self.warmup:
            return False
        if current_epoch + 1 == self.warmup and self.warmup != 0:
            if self.verbose:
                print("EarlyStopping Warmup ended")
        if self.best_loss - val_loss > self.epsilon:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
        
# Paper: https://proceedings.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf 
# NB: Implemented without truncation 
class GCELoss(nn.Module):
    def __init__(self, q: float = 0.7, reduction: str = "mean", *args, **kwargs) -> None:
        if reduction not in {"mean", "none", "sum"}:
            raise ValueError(f"Unsupported reduction parameter {reduction}. Use one among ['mean', 'sum', 'none'] passed as 'string' type argument.")

        super().__init__(*args, **kwargs)
        self.reduction: str = reduction
        self.q: float = q

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  
        outputs = outputs.double() 
        softmax_probs : torch.Tensor = F.softmax(outputs, dim=1)
        gathered_preds: torch.Tensor = torch.gather(softmax_probs, dim=1, index=torch.unsqueeze(targets, 1))
        unreduced_loss: torch.Tensor = (1 - gathered_preds ** self.q) / self.q
        
        match self.reduction:
            case "mean": return unreduced_loss.mean()
            case "sum" : return unreduced_loss.sum()
            case "none": return unreduced_loss
            
    def __repr__(self) -> str:
        return f"GCELoss(q={self.q}, reduction={self.reduction})"

def collect_preds_from_features(model, features, labels, bias_labels):
    preds = []
    targets = []
    btargets = []
    
    model.eval()
    with torch.no_grad():
        for x, y, b in zip(features, labels, bias_labels):
            x = torch.from_numpy(x).unsqueeze(0)
            y = torch.as_tensor((y, )).long()
            b = torch.as_tensor((b, )).long()
            preds.append(make_prediction(model, x))
            targets.append(y)
            btargets.append(b)

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(btargets, axis=0)
    )

def make_prediction(m, x: torch.Tensor) -> np.ndarray:
        _, pred = torch.max(m.head(x.to(m.device)), dim=1)
        return pred.cpu().numpy()
