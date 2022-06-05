import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, in_feats, n_classes):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(in_feats, 8)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(8, 64)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(64, n_classes)

    def forward(self, x):
        out = self.linear(x)
        x = self.act1(out)
        x = self.l2(x)
        x = self.act2(x)
        return self.l3(x)
