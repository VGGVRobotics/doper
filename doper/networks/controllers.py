__all__ = ["LinearReLUController"]

import torch
import torch.nn as nn


class LinearReLUController(nn.Module):
    def __init__(self, model_config):
        super().__init__(self)
        input_dim = model_config['input_dim']
        internal_dim = model_config['internal_dim']
        output_dim = model_config['output_dim']

        self.controller = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.ReLU(inplace=True),
            nn.Linear(internal_dim, 2 * internal_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * internal_dim, output_dim),
        )

    def forward(self, controller_input):
        return self.controller(controller_input)
