import torch
from torch import nn


class JointLoss(nn.Module):
    def __init__(self, r=0.5):
        super(JointLoss, self).__init__()
        self.forecast_criterion = nn.MSELoss()
        self.reconstruct_criterion = nn.MSELoss()
        self.r = r

    def forward(self, x, y, reconstruct, forecast):
        forecast_loss = torch.sqrt(self.forecast_criterion(forecast, y))

        reconstruct_loss = torch.sqrt(self.reconstruct_criterion(reconstruct, x))

        loss = self.r * forecast_loss + (1 - self.r) * reconstruct_loss

        return forecast_loss, reconstruct_loss, loss
