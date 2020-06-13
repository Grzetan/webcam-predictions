import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
        def __init__(self, outputs):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
            self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
            self.lin1 = nn.Linear(in_features=24*13*18, out_features=256)
            self.lin2 = nn.Linear(in_features=256, out_features=128)
            self.lin3 = nn.Linear(in_features=128, out_features=64)
            self.out = nn.Linear(in_features=64, out_features=outputs)
            self.dropout = nn.Dropout(0.2)

        def forward(self, t):
            t = self.conv1(t.float())
            t = F.relu(t)
            t = F.max_pool2d(t, kernel_size=2, stride=2)

            t = self.conv2(t)
            t = F.relu(t)
            t = F.max_pool2d(t, kernel_size=2, stride=2)
            
            t = self.conv3(t)
            t = F.relu(t)
            t = F.max_pool2d(t, kernel_size=2, stride=2)
            
            t = t.reshape(-1, 24*13*18)

            t = self.lin1(t)
            t = F.relu(t)
            t = self.dropout(t)

            t = self.lin2(t)
            t = F.relu(t)
            t = self.dropout(t)

            t = self.lin3(t)
            t = F.relu(t)
            t = self.dropout(t)

            t = self.out(t)

            return t