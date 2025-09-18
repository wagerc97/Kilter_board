import torch
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, path="best_model.pt"):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def load_best(self, model, device="cpu"):
        model.load_state_dict(torch.load(self.path))
        model.to(device)
        return model



def plot_training_history(history):
    """
    Plot training and test loss/accuracy from history dict.

    Args:
        history: dict with keys 'train_loss', 'test_loss', 
                'train_acc', 'test_acc'
    """
    epochs = range(len(history["train_loss"]))

    plt.figure(figsize=(12, 5))

    # ---- Loss plot ----
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()

    # ---- Accuracy plot ----
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()