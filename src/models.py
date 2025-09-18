import torch.nn as nn


class ShallowMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, drop_out):
        super(ShallowMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, num_classes)   # classification head
    )


    def forward(self, x):
        return self.model(x)