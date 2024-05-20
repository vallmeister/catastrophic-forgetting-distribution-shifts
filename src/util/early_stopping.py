import torch


class EarlyStopping:
    def __init__(self, path, model, patience=10):
        self.patience = patience
        self.path = path
        self.model = model
        self.counter = 0

        self.val_acc_max = -1.0

    def __call__(self, val_acc):
        acc_worse = val_acc <= self.val_acc_max
        if acc_worse:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            if not acc_worse:
                self.val_acc_max = val_acc
            self.counter = 0
            self.save_checkpoint()
        return False

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), str(self.path))

    def __del__(self):
        if self.path.is_file():
            self.path.unlink()
