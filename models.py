import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

"""Has to output 136 values, that will be coordinates of the facial keypoints"""
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)
        
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.fc2 = nn.Linear(1000, 136)
        
        
    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
