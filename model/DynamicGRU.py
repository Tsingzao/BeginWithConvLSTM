import torch
import torch.nn as nn
import torch.nn.functional as F


class QuasiAtt(nn.Module):
    def __init__(self, channel, K):
        super(QuasiAtt, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel//4, bias=False)
        self.fc2 = nn.Linear(channel//4, K, bias=False)

    def forward(self, x):
        x = self.avg(x)
        x = F.relu(self.fc1(x.view(x.shape[0], x.shape[1])))
        x = F.softmax(self.fc2(x),1)
        return x

