from typing import List, Sequence

import jax
import jax.numpy as np
import numpy as onp
import torch


def input_to_pytorch(controller_inputs: List[Sequence]) -> torch.Tensor:
    """
    Converts a list of numpy.ndarrays or lists into pytorch tensor
    Args:
        controller_inputs: list of lists or numpy.ndarrays

    Returns:
        pytorch tensor suitable for usage as input to the controller net
    """
    controller_input = torch.cat(
        [
            torch.from_numpy(onp.array(sensor_reading).astype(onp.float32)).view(1, -1)
            for sensor_reading in controller_inputs
        ],
        dim=1,
    )
    return controller_input


def jax_grads_to_pytorch(jax_grads: jax.interpreters.xla.DeviceArray) -> torch.Tensor:
    """
    Converts gradient from jax simulation to pytorch tensor
    Args:
        jax_grads: gradients from jax

    Returns:
        torch.Tensor with gradients from jax
    """
    return torch.from_numpy(onp.array(jax_grads))


def pytorch_to_jax(torch_tensor: torch.Tensor) -> jax.interpreters.xla.DeviceArray:
    return np.array(torch_tensor.cpu().data.numpy())
