from torch import nn


class ForecastModel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(ForecastModel, self).__init__()
        self.in_dim = in_dim
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)

        output = self.fc(output)
        return output
