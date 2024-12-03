import torch.nn as nn


class LSSRNLoss(nn.Module):
    def __init__(self):
        super(LSSRNLoss, self).__init__()
        
    def forward(self, y_pred, y):
        return nn.functional.mse_loss(y_pred, y)
