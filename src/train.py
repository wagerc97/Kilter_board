import sys
import argparse
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

# Project modules
from utils import *
from utils_visualisation import *
from Data_Handler import *
from models import *
from engine import *

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassAccuracy
from torchinfo import summary




# Image filters
import kornia.filters as KF


#-------------- Hyperparameters ---------------

#data Transformation
blur_kernel_size = 3 
blur_sigma = 1.0
map = True
max_samples= 36000
label_filter=[5, 6]

#model architecture
hidden_units_CNN = 16
hidden_units_classifier = 16

#Optimization
lr = 1e-4
weight_decay = 1e-4

num_epochs = 5
patience = 3
batch_size = 32

model_type = "CNN_K3"
model_name = "tmp_model"

# -------------- Argparse setup ---------------
parser = argparse.ArgumentParser()

parser.add_argument("--blur-kernel-size", type=int, default=blur_kernel_size)
parser.add_argument("--blur-sigma", type=float, default=blur_sigma)
parser.add_argument("--map",  default=map)  # bool handled like this
parser.add_argument("--max-samples", type=int, default=max_samples)
parser.add_argument("--label-filter", type=list[int], default=label_filter)

parser.add_argument("--hidden-units-cnn", type=int, default=hidden_units_CNN)
parser.add_argument("--hidden-units-classifier", type=int, default=hidden_units_classifier)

parser.add_argument("--lr", type=float, default=lr)
parser.add_argument("--weight-decay", type=float, default=weight_decay)

parser.add_argument("--num-epochs", type=int, default=num_epochs)
parser.add_argument("--patience", type=int, default=patience)
parser.add_argument("--batch-size", type=int, default=batch_size)

parser.add_argument("--model-name", type=str, default=model_name)
parser.add_argument("--model-type", type=str, default=model_type)

args = parser.parse_args()

# Pick the class based on CLI argument
ModelClass = CNN_MODEL_REGISTRY[args.model_type]
# ----------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader = CNN_dataloaders(
    board_names=["12 x 12 with kickboard Square"],
    max_samples=36000,
    label_filter = args.label_filter ,     # 6a+ & 6b
    blur_kernel_size=args.blur_kernel_size,
    blur_sigma=args.blur_sigma,
    map=args.map,
    batch_size=args.batch_size,
    train_test_split=0.8
)

X, y = next(iter(train_dataloader))
num_classes = y.shape[1]
num_features = X.shape[1]

loss_fn = torch.nn.CrossEntropyLoss()
acc_fn = Multiclass_accuracy()

kwargs = {
    "input_shape": num_features,                # from dataloader
    "hidden_units_CNN": args.hidden_units_cnn,  # from CLI
    "hidden_units_classifier": args.hidden_units_classifier,
    "output_shape": num_classes                 # from dataloader
}

model = ModelClass(**kwargs).to(device)

"""model = CNN_K9(input_shape= num_features,
                     hidden_units_CNN= hidden_units_CNN, 
                     hidden_units_classifier=  hidden_units_classifier, 
                     output_shape=num_classes 
                     ).to(device)"""
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

trained_model, history = train_model(
        model=model,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        acc_fn=acc_fn,
        device=device,
        epochs=args.num_epochs,      # you can change this
        patience=args.patience    # early stopping patience
    )

save_model(model=model, model_name=args.model_name)


