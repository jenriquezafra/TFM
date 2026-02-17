import torch
from pathlib import Path

################################## function to save checkpoints
def save_checkpoints(
        path: Path,
        model,
        optimizer,
        epoch: int,
        train_loss: float,
        val_loss: float,
        loss_name: str,
):
    
    info = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "loss_name": loss_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(info, path)

################################## class for Early Stopping
class EarlyStopping:
    def __init__(
            self,
            patience: int,
            min_delta: float = 0.0,
            warmup_epochs: int = 0,
            mode: str = "min"
    ):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.mode = mode.lower()

        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        
        self.best = float("inf") if self.mode == "min" else -float("inf")
        self.counter = 0

    def step(self, epoch: int, value: float) -> bool:
        """
        Returns True if training should stop
        """
        # dont stop on the warmup
        if epoch <= self.warmup_epochs:
            self._update_best(value)
            return False

        improved = self._is_improvement(value)

        if improved:
            self.best = value
            self.counter = 0
            return False
        
        self.counter += 1
        return self.counter >= self.patience
    

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        else:
            return value > (self.best + self.min_delta)
        

    def _update_best(self, value: float) -> None:
        # on warmup, update without taking care of the min_delta
        if (self.mode == "min" and value < self.best) or (self.mode == "max" and value > self.best):
            self.best = value
            self.counter = 0


################################## StepLR ###############################
def build_step_lr(optimizer, step_size: int, gamma: float):
    """
    Step Learning Rate Scheduler
    Decreases the learning rate by a factor of gamma every step_size epochs
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )
