import torch
import torch.nn as nn

import config.config as config
from layers.TCN import TCN


class EnhancedEncoder(nn.Module):
    def __init__(self, hidden_size=config.hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.tcn_intemp = nn.Sequential(
            PermuteLayer(),
            TCN(num_inputs=1,
                num_channels=[hidden_size // 2, hidden_size],
                kernel_size=4),
            PermuteLayer()
        )

        self.tcn_outtemp = nn.Sequential(
            PermuteLayer(),
            TCN(num_inputs=1,
                num_channels=[hidden_size // 2, hidden_size],
                kernel_size=4),
            PermuteLayer()
        )

        self.tcn_power = nn.Sequential(
            PermuteLayer(),
            TCN(num_inputs=1,
                num_channels=[hidden_size // 2, hidden_size],
                kernel_size=4),
            PermuteLayer()
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size * 3,
                nhead=4,
                dim_feedforward=hidden_size * 3 * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )

        self.output_proj = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, x):
        intemp = x[:, :, 0:1]
        outtemp = x[:, :, 1:2]
        power = x[:, :, 2:3]

        power_zeroed = power.clone()
        power_zeroed[:, -1:, :] = 0

        tcn_intemp = self.tcn_intemp(intemp)
        tcn_outtemp = self.tcn_outtemp(outtemp)
        tcn_power = self.tcn_power(power_zeroed)

        q = torch.cat([tcn_intemp, tcn_outtemp, tcn_power], dim=-1)

        output = self.transformer(q)

        r = self.output_proj(output)
        return r


class PermuteLayer(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1) if x.dim() == 3 else x
