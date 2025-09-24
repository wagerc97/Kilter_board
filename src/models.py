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
                                kernel_size= 5,
                                stride=1,
                                padding=2),
                        nn.BatchNorm2d(hidden_units_CNN),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=hidden_units_CNN,
                                out_channels= hidden_units_CNN,
                                kernel_size= 3,
                                stride=1,
                                padding=2),
                        nn.BatchNorm2d(hidden_units_CNN),
                        nn.ReLU(),
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
                #print(x.shape)
                x = self.conv_block_1(x)

                x = self.classifier(x)
                return x
        
        
        