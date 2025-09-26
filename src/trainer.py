from tqdm.auto import tqdm
import torch.optim as optim
from timeit import default_timer as timer
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torchmetrics import Metric
import shutil
import time

from utils import EarlyStopping


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: _Loss,
    optimizer: Optimizer,
    acc_fn: Metric,
    device: torch.device,
    epochs: int = 1000,
    patience: int = 5,
    checkpoint_path: str = "best_model.pt"
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train a PyTorch model with early stopping.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test/validation data.
        loss_fn: Loss function (e.g. nn.CrossEntropyLoss).
        optimizer: Optimizer (e.g. torch.optim.Adam).
        acc_fn: Accuracy function/metric (e.g. MulticlassAccuracy).
        device: Device to train on (torch.device('cpu') or torch.device('cuda')).
        epochs: Maximum number of epochs to train.
        patience: Patience for early stopping.
        checkpoint_path: File path to save the best model.

    Returns:
        model: The trained model loaded with best weights.
        history: Dict with training/test loss and accuracy:
            {
                "train_loss": [...],
                "train_acc": [...],
                "test_loss": [...],
                "test_acc": [...]
            }
    """
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)

    all_train_loss, all_train_acc = [], []
    all_test_loss, all_test_acc = [], []


    for epoch in (range(epochs)):
        # --- training ---
        
        model.train()
        acc_fn.reset()
        train_loss, train_acc = 0.0, 0.0
        t0 = time.perf_counter()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            acc = acc_fn(torch.argmax(y_pred, dim=1),
                        torch.argmax(y, dim=1)).item()

            train_loss += loss.item()
            train_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
        
        t1 = time.perf_counter()

        # --- evaluation ---
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.inference_mode():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                test_pred = model(X_test)

                test_loss += loss_fn(test_pred, y_test).item()
                test_acc += acc_fn(torch.argmax(test_pred, dim=1),
                                torch.argmax(y_test, dim=1)).item()

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)
        
        t2 = time.perf_counter()

        #if epoch % 1 == 0:
            #print(f"\nEpoch {epoch:04d} | "
            #        f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f} | "
            #        f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")
            #print(f"Timing: total {t2 - t0:.3f} s | "
            #        f"train {t1 - t0:.3f} s | "
            #        f"eval {t2 - t1:.3f} s")
        # --- early stopping ---
        if early_stopping(test_loss, model):
            #print(f"Early stopping at epoch {epoch}")
            #print(f"Epoch {epoch:04d} | "
            #        f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.5f} | "
            #        f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")
            break

    # 1. Move "best_model.pt" to checkpoint
    #shutil.copy("best_model.pt", checkpoint_path)


    history = {
        "train_loss": all_train_loss,
        "train_acc": all_train_acc,
        "test_loss": all_test_loss,
        "test_acc": all_test_acc
    }

    return model, history
