__all__ = ["UndirectedLinearReLUController"]

import torch.nn as nn


class UndirectedLinearReLUController(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        input_dim = model_config["input_dim"]
        hidden_dim = model_config["hidden_dim"]
        output_dim = model_config["output_dim"]

        self.controller = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, output_dim),
        )

    def forward(self, controller_input):
        return self.controller(controller_input)


class DirectedLinearReLUController(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        input_dim = model_config["input_dim"]
        hidden_dim = model_config["hidden_dim"]
        output_dim = model_config["output_dim"]

        self.controller = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, output_dim),
        )

    def forward(self, controller_input):
        return self.controller(controller_input)
