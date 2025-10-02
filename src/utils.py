import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience=5, path="checkpoints/best_model.pt"):
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


    

def get_grade(idx: int) -> str | None:
    """
    Map a numeric class index to a climbing grade string.

    Args:
        idx (int): Index in the grade list (0-based).
                   Must be between 0 and 19 inclusive.

    Returns:
        str | None: The corresponding climbing grade (e.g., "6a+", "7b") 
        if idx is valid, otherwise None.
    """
    grades = [
        "4c", "5a", "5b", "5c",
        "6a", "6a+", "6b", "6b+", "6c", "6c+",
        "7a", "7a+", "7b", "7b+", "7c", "7c+",
        "8a", "8a+", "8b", "8b+", "8c"
    ]
    return grades[idx] if 0 <= idx < len(grades) else None




def measure_loader_time(loader: DataLoader, num_batches: int = 20, skip_first: bool = True) -> float:
    """
    Measure average batch loading time for a PyTorch DataLoader.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        The DataLoader to benchmark.
    num_batches : int, default=20
        Number of batches to time. If the loader has fewer batches,
        iteration stops at the end of the dataset.
    skip_first : bool, default=True
        Whether to exclude the first batch from the average,
        since it often includes extra setup overhead.

    Returns
    -------
    float
        Average batch loading time in milliseconds.
    """
    loader_iter = iter(loader)
    times = []

    for i in range(num_batches):
        try:
            start = time.perf_counter()
            X, y = next(loader_iter)
            end = time.perf_counter()
        except StopIteration:
            break

        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        #print(f"Batch {i}: {elapsed_ms:.2f} ms")

    if skip_first and len(times) > 1:
        times = times[1:]

    avg_time = sum(times) / len(times)
    print(f"\nAverage batch load time: {avg_time:.2f} ms over {len(times)} batches")
    return avg_time


class Multiclass_accuracy:
    def reset(self):
        pass  # no state to reset
    
    def __call__(self, y_pred, y_true):
        return multiclass_accuracy(y_pred, y_true)

def multiclass_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute multiclass accuracy.
    
    Parameters
    ----------
    y_pred : torch.Tensor
        Model outputs, shape [N, num_classes], raw logits or probabilities.
    y_true : torch.Tensor
        Ground truth labels, shape [N], as integer class indices.
    
    Returns
    -------
    float
        Accuracy = correct / total.
    """
    # predicted class = argmax over class dimension
    correct = (y_pred == y_true).sum()
    total = y_true.size(0)
    return correct / total


def save_model(model:nn.Module, model_name=str):
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)  # make sure folder exists

    model_path = models_dir / f"{model_name}.pt"

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")