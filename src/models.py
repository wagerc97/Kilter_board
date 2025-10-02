import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F













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
    
    
    
class shallowCNN(nn.Module):
        def __init__(self,
                input_shape: int,
                hidden_units_CNN: int,
                hidden_units_classifier: int,
                output_shape: int,
                drop_out: int = 0.001
                ):
                super().__init__()
                self.conv_block_1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_shape,
                                out_channels= hidden_units_CNN,
                                kernel_size= 3,
                                stride=1,
                                padding=1),
                        nn.ReLU(),                        
                        nn.BatchNorm2d(hidden_units_CNN),
                        #nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=hidden_units_CNN,
                                out_channels= hidden_units_CNN,
                                kernel_size= 3,
                                stride=1,
                                padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(hidden_units_CNN)
                        #nn.MaxPool2d(kernel_size=2)
                )
                self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.LazyLinear(out_features=hidden_units_classifier),
                        nn.ReLU(),
                        nn.Dropout(drop_out),
                        nn.Linear(hidden_units_classifier, hidden_units_classifier),
                        nn.ReLU(),
                        nn.Dropout(drop_out),
                        nn.Linear(hidden_units_classifier, output_shape)
                )
        def forward(self, x):
                x = self.conv_block_1(x)
                x = self.classifier(x)
                return x
        
        
class CNN_K3(nn.Module):
        def __init__(self,
                input_shape: int,
                hidden_units_CNN: int,
                hidden_units_classifier: int,
                output_shape: int,
                drop_out: int = 0.001
                ):
                super().__init__()
                self.conv_block_1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_shape,
                                out_channels= hidden_units_CNN,
                                kernel_size= 3,
                                stride=1,
                                padding=1),
                        nn.ReLU(),                        
                        nn.BatchNorm2d(hidden_units_CNN),
                        #nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=hidden_units_CNN,
                                out_channels= hidden_units_CNN,
                                kernel_size= 3,
                                stride=1,
                                padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(hidden_units_CNN)
                        #nn.MaxPool2d(kernel_size=2)
                )
                self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.LazyLinear(out_features=output_shape),
                        nn.ReLU()
                )
        def forward(self, x):
                x = self.conv_block_1(x)
                x = self.classifier(x)
                return x
        
class CNN_K5(nn.Module):
        def __init__(self,
                input_shape: int,
                hidden_units_CNN: int,
                hidden_units_classifier: int,
                output_shape: int,
                drop_out: int = 0.001
                ):
                super().__init__()
                self.conv_block_1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_shape,
                                out_channels= hidden_units_CNN,
                                kernel_size= 5,
                                stride=1,
                                padding=2),
                        nn.ReLU(),                        
                        nn.BatchNorm2d(hidden_units_CNN),
                        #nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=hidden_units_CNN,
                                out_channels= hidden_units_CNN,
                                kernel_size= 5,
                                stride=1,
                                padding=2),
                        nn.ReLU(),
                        nn.BatchNorm2d(hidden_units_CNN)
                        #nn.MaxPool2d(kernel_size=2)
                )
                self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.LazyLinear(out_features=output_shape),
                        nn.ReLU()
                )
        def forward(self, x):
                return self.classifier(self.conv_block_1(x))
                
        
class CNN_K7(nn.Module):
        def __init__(self,
                input_shape: int,
                hidden_units_CNN: int,
                hidden_units_classifier: int,
                output_shape: int,
                drop_out: int = 0.001
                ):
                super().__init__()
                self.conv_block_1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_shape,
                                out_channels= hidden_units_CNN,
                                kernel_size= 7,
                                stride=1,
                                padding=3),
                        nn.ReLU(),                        
                        nn.BatchNorm2d(hidden_units_CNN),
                        #nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=hidden_units_CNN,
                                out_channels= hidden_units_CNN,
                                kernel_size= 5,
                                stride=1,
                                padding=2),
                        nn.ReLU(),
                        nn.BatchNorm2d(hidden_units_CNN)
                        #nn.MaxPool2d(kernel_size=2)
                )
                self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.LazyLinear(out_features=output_shape),
                        nn.ReLU()
                )
        def forward(self, x):
                x = self.conv_block_1(x)
                x = self.classifier(x)
                return x

        
class CNN_K9(nn.Module):
        def __init__(self,
                input_shape: int,
                hidden_units_CNN: int,
                hidden_units_classifier: int,
                output_shape: int,
                drop_out: int = 0.001
                ):
                super().__init__()
                self.conv_block_1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_shape,
                                out_channels= hidden_units_CNN,
                                kernel_size= 9,
                                stride=1,
                                padding=4),
                        nn.ReLU(),                        
                        nn.BatchNorm2d(hidden_units_CNN),
                        #nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=hidden_units_CNN,
                                out_channels= hidden_units_CNN,
                                kernel_size= 5,
                                stride=1,
                                padding=2),
                        nn.ReLU(),
                        nn.BatchNorm2d(hidden_units_CNN)
                        #nn.MaxPool2d(kernel_size=2)
                )
                self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.LazyLinear(out_features=hidden_units_classifier),
                        nn.ReLU(),
                        nn.Dropout(drop_out),
                        nn.Linear(hidden_units_classifier, hidden_units_classifier),
                        nn.ReLU(),
                        nn.Dropout(drop_out),
                        nn.Linear(hidden_units_classifier, output_shape)
                )
        def forward(self, x):
                x = self.conv_block_1(x)
                x = self.classifier(x)
                return x
        
        
class CNN_K11(nn.Module):
        def __init__(self,
                input_shape: int,
                hidden_units_CNN: int,
                hidden_units_classifier: int,
                output_shape: int,
                drop_out: int = 0.001
                ):
                super().__init__()
                self.conv_block_1 = nn.Sequential(
                        nn.Conv2d(in_channels=input_shape,
                                out_channels= hidden_units_CNN,
                                kernel_size= 11,
                                stride=1,
                                padding=5),
                        nn.ReLU(),                        
                        nn.BatchNorm2d(hidden_units_CNN),
                        #nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=hidden_units_CNN,
                                out_channels= hidden_units_CNN,
                                kernel_size= 5,
                                stride=1,
                                padding=2),
                        nn.ReLU(),
                        nn.BatchNorm2d(hidden_units_CNN)
                        #nn.MaxPool2d(kernel_size=2)
                )
                self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.LazyLinear(out_features=hidden_units_classifier),
                        nn.ReLU(),
                        nn.Dropout(drop_out),
                        nn.Linear(hidden_units_classifier, hidden_units_classifier),
                        nn.ReLU(),
                        nn.Dropout(drop_out),
                        nn.Linear(hidden_units_classifier, output_shape)
                )
        def forward(self, x):
                x = self.conv_block_1(x)
                x = self.classifier(x)
                return x
        

class CNN_PeRo(nn.Module):
    def __init__(self,
                input_shape: int = 4,
                hidden_units_CNN: int = 32,
                hidden_units_classifier: int= 32,
                output_shape: int = 32
                ):
        super().__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=hidden_units_CNN, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(hidden_units_CNN)
        
        self.conv2 = nn.Conv2d(hidden_units_CNN, hidden_units_CNN, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(hidden_units_CNN)
        
        self.conv3 = nn.Conv2d(hidden_units_CNN, hidden_units_CNN * 2, kernel_size=3)
        self.bn3   = nn.BatchNorm2d(hidden_units_CNN * 2)
        
        self.conv4 = nn.Conv2d(hidden_units_CNN * 2, hidden_units_CNN * 2, kernel_size=3)
        self.bn4   = nn.BatchNorm2d(hidden_units_CNN * 2)
        
        # Fully connected layers
        self.fc1 = nn.LazyLinear(out_features=hidden_units_classifier)  # adjust dimensions after convs
        self.fc2 = nn.Linear(hidden_units_classifier, output_shape)  # linear output

    def forward(self, x):
        # Input: (batch, 18*11) → reshape to (batch, 1, 18, 11)
        #x = x.view(-1, 1, 18, 11)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.flatten(x, 1)  # flatten all dims except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # linear activation
        return x
  
CNN_MODEL_REGISTRY = {
    "CNN_PeRo": CNN_PeRo,
    "CNN_K3": CNN_K3,
    "CNN_K5": CNN_K5,
    "CNN_K7": CNN_K7,
    "CNN_K9": CNN_K9,
    "CNN_K11": CNN_K11,
    "shallowCNN": shallowCNN,
    
}

