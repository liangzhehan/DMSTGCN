import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, normalization, lrate, wdecay, device, days=288,
                 dims=40, order=2):
        self.model = DMSTGCN(device, num_nodes, dropout, out_dim=seq_length, residual_channels=nhid,
                             dilation_channels=nhid, end_channels=nhid * 16, days=days, dims=dims, order=order,
                             in_dim=in_dim, normalization=normalization)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, ind):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input, ind)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return mae, mape, rmse

    def eval(self, input, real_val, ind):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input, ind)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return mae, mape, rmse
