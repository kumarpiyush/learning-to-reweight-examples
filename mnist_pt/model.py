import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module) :
    def __init__(self) :
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)

    def forward(self, x) :
        tval = self.conv1(x)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = self.conv2(tval)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = self.conv3(tval)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)