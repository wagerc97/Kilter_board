import torch
from torchmetrics import Metric
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from utils import *
import torch
import warnings
from collections import Counter, defaultdict






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


def plot_training_histories(histories):
    """
    Plot training and test loss/accuracy for multiple models.
    Train and test curves for the same model use the same color.
    
    Args:
        histories: dict of {model_name: history_dict}, 
                   where each history_dict has keys 
                   'train_loss', 'test_loss', 'train_acc', 'test_acc'
    """
    plt.figure(figsize=(14, 6))

    # ---- Loss plot ----
    plt.subplot(1, 2, 1)
    for idx, (model_name, history) in enumerate(histories.items()):
        epochs = range(len(history["train_loss"]))
        color = f"C{idx}"  # use matplotlib's default color cycle
        plt.plot(epochs, history["train_loss"], label=f"{model_name} Train", color=color)
        plt.plot(epochs, history["test_loss"], linestyle="--", label=f"{model_name} Test", color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()

    # ---- Accuracy plot ----
    plt.subplot(1, 2, 2)
    for idx, (model_name, history) in enumerate(histories.items()):
        epochs = range(len(history["train_acc"]))
        color = f"C{idx}"
        plt.plot(epochs, history["train_acc"], label=f"{model_name} Train", color=color)
        plt.plot(epochs, history["test_acc"], linestyle="--", label=f"{model_name} Test", color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    
    
def plot_sample_routes(dataset, n: int = 10, grades=None):
    """
    Plot n sample routes from a dataset using `plot_route`.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset returning (route_tensor, label).
    n : int, default=10
        Number of routes to plot.
    grades : list[str], optional
        List of grade labels for mapping class indices to human-readable grades.
    """
    from random import sample

    # Pick n random indices
    idxs = sample(range(len(dataset)), min(n, len(dataset)))

    for i in idxs:
        X, y = dataset[i]  # X = route_tensor, y = one-hot or label
        # Handle one-hot labels
        label_idx = int(torch.argmax(y).item()) if y.ndim > 0 else int(y)
        plot_route(X, label_idx=label_idx, title=f"Route {i}", grades=grades)


def plot_route(route_tensor, label_idx=None, title="Route", grades=None):
    """
    Plot a climbing route from a 4xHxW tensor.
    
    Args:
        route_tensor (torch.Tensor): shape [4, H, W] or [1, 4, H, W].
                                     Channels = [start, middle, foot, finish].
        label_idx (int, optional): difficulty class index (0–N).
        title (str): plot title.
        grades (list[str], optional): list of grade labels for indexing.
    """
    # Squeeze if batch dim exists
    if route_tensor.dim() == 4:  # [1, 4, H, W]
        route_tensor = route_tensor.squeeze(0)

    if route_tensor.shape[0] != 4:
        raise ValueError(f"Expected [4, H, W], got {route_tensor.shape}")

    H, W = route_tensor.shape[1], route_tensor.shape[2]

    # Build RGB image
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Define colors per channel
    colors = {
        "start":  (1.0, 0.0, 0.0),  # red
        "middle": (0.0, 1.0, 0.0),  # green
        "foot":   (0.0, 0.0, 1.0),  # blue
        "finish": (1.0, 1.0, 0.0),  # yellow
    }

    for i, key in enumerate(colors.keys()):
        mask = route_tensor[i].cpu().numpy()
        for c in range(3):
            img[..., c] += mask * colors[key][c]

    img = np.clip(img, 0, 1)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin="lower", interpolation="nearest")

    # Title with optional label
    if label_idx is not None:
        plt.title(f"{title} (Label: {get_grade(label_idx)})")

    plt.xlabel("x")
    plt.ylabel("y")

    # Legend
    patches = [mpatches.Patch(color=colors[name], label=name) for name in colors.keys()]
    plt.legend(handles=patches, loc="upper right")

    plt.show()
    


def plot_label_distribution(train_dataset, test_dataset, num_classes):
    """
    Plot label distribution for train and test datasets.

    Prints counts and percentages per class, and plots the counts as bars.
    """
    def collect_labels(subset):
        ds = subset.dataset        # underlying dataset
        idxs = subset.indices      # chosen indices
        labels = []
        for i in idxs:
            _, y = ds[i]
            labels.append(y)
        return torch.stack(labels)

    # collect counts
    train_labels = collect_labels(train_dataset)
    test_labels = collect_labels(test_dataset)

    train_counts = train_labels.sum(dim=0).numpy()
    test_counts = test_labels.sum(dim=0).numpy()

    # print distributions
    total_train = train_counts.sum()
    total_test = test_counts.sum()

    print("Train distribution:")
    for i, count in enumerate(train_counts):
        pct = 100 * count / total_train if total_train > 0 else 0
        if pct > 0.1:
            print(f"  Class {get_grade(i)}: {count} ({pct:.1f}%)")

    print("\nTest distribution:")
    for i, count in enumerate(test_counts):
        pct = 100 * count / total_test if total_test > 0 else 0
        if pct > 0.1:
            print(f"  Class {get_grade(i)}: {count} ({pct:.1f}%)")

    # plot counts only
    x = range(num_classes)
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], train_counts, width=width, label="Train")
    plt.bar([i + width/2 for i in x], test_counts, width=width, label="Test")

    plt.xlabel("Difficulty class index")
    plt.ylabel("Count")
    plt.title("Label occurrence in train and test sets")
    plt.legend()
    plt.show()
    




import torch
import matplotlib.pyplot as plt

def plot_prediction_distribution(model, test_loader, num_classes, model_name, acc_fn, device="cpu"):
    """
    Plot true vs predicted label distribution for a given model on test_loader,
    and print overall accuracy.

    Args:
        model: trained torch.nn.Module
        test_loader: DataLoader for the test set
        num_classes: number of output classes
        model_name: string name of the model (for plot title)
        acc_fn: metric function (e.g., MulticlassAccuracy from torchmetrics)
        device: torch device (default 'cpu')
    """
    model.eval()
    model.to(device)

    preds, labels = [], []
    acc_total = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            batch_preds = torch.argmax(logits, dim=1)   # predicted class indices
            batch_labels = torch.argmax(y, dim=1)       # true class indices

            preds.append(batch_preds.cpu())
            labels.append(batch_labels.cpu())

            # compute batch accuracy
            acc_total += acc_fn(batch_preds, batch_labels).item()

    # flatten lists into tensors
    preds = torch.cat(preds)    # shape (N,)
    labels = torch.cat(labels)  # shape (N,)

    total = len(labels)
    correct = (preds == labels).sum().item()
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    acc_avg = 100.0 * acc_total / len(test_loader)

    # --- Distributions ---
    true_counts = torch.bincount(labels, minlength=num_classes).numpy()
    pred_counts = torch.bincount(preds, minlength=num_classes).numpy()

    print(f"\n📊 Distributions for {model_name}:")
    print("True labels:")
    for i, count in enumerate(true_counts):
        pct = 100 * count / total if total > 0 else 0
        if pct > 0.1:
            print(f"  Class {i}: {count} ({pct:.1f}%)")

    print("\nPredicted labels:")
    for i, count in enumerate(pred_counts):
        pct = 100 * count / total if total > 0 else 0
        if pct > 0.1:
            print(f"  Class {i}: {count} ({pct:.1f}%)")

    print(f"\n✅ Accuracy (overall): {accuracy:.2f}% ({correct}/{total})")
    print(f"ℹ️  Accuracy via acc_fn (avg over batches): {acc_avg:.2f}%")

    # ---- Plot ----
    x = range(num_classes)
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], true_counts, width=width, label="True")
    plt.bar([i + width/2 for i in x], pred_counts, width=width, label="Predicted")

    plt.xticks(ticks=list(x), labels=[get_grade(i) for i in x], rotation=45)
    plt.ylabel("Count")
    plt.title(f"True vs Predicted Distribution - {model_name}\nAccuracy: {accuracy:.2f}%")
    plt.legend()
    plt.show()






def print_hold_statistics(dataset, num_classes):
    """
    Print average hold counts and role composition per difficulty class.

    Warnings
    --------
    - Warns if any route does not contain at least one 'start' and one 'finish' hold.
    
    Parameters
    ----------
    dataset : torch.utils.data.Subset
        Subset wrapping your climbing dataset (with route_groups access).
    num_classes : int
        Number of difficulty classes.
    get_grade : callable, optional
        Maps class index -> human-readable label (e.g. V-grades).
    """
    ds = dataset.dataset
    idxs = dataset.indices

    class_hold_counts = defaultdict(list)  # difficulty -> [total holds]
    class_roles = defaultdict(list)        # difficulty -> list of role counters

    for i in idxs:
        route_id = ds.unique_routs[i]
        route_df = ds.route_groups[route_id]
        difficulty = int(route_df["difficulty"].unique()[0])

        role_counts = Counter(route_df["role_name"])
        total_holds = len(route_df)

        # warn if no start or no finish
        if role_counts.get("start", 0) < 1 or role_counts.get("finish", 0) < 1:
            warnings.warn(
                f"Route {route_id} (class {get_grade(difficulty)}) "
                f"has start={role_counts.get('start', 0)} and "
                f"finish={role_counts.get('finish', 0)}"
            )

        class_hold_counts[difficulty].append(total_holds)
        class_roles[difficulty].append(role_counts)

    # print per-class stats
    print("Hold statistics per difficulty:\n")
    for cls in range(num_classes):
        if cls not in class_hold_counts:
            continue

        avg_total = sum(class_hold_counts[cls]) / len(class_hold_counts[cls])

        role_sum = Counter()
        for rc in class_roles[cls]:
            role_sum.update(rc)
        avg_roles = {role: role_sum[role] / len(class_roles[cls]) for role in ["start", "middle", "foot", "finish"]}

        print(f"Class {get_grade(cls)}:")
        print(f"  Avg total holds: {avg_total:.2f}")
        print(f"  Avg role composition: "
              f"start={avg_roles['start']:.2f}, "
              f"middle={avg_roles['middle']:.2f}, "
              f"foot={avg_roles['foot']:.2f}, "
              f"finish={avg_roles['finish']:.2f}")
        print()


