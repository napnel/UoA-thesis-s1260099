import numpy as np
import gym
import torch.nn as nn
from typing import Union, Tuple, Any, List

from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict, Any
from ray.rllib.models.utils import get_activation_fn


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        initializer: Any = None,
        activation_fn: Any = None,
        use_bias: bool = True,
        bias_init: float = 0.0,
        use_batch_norm: bool = False,
    ):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size(int): Input size for FC Layer
            out_size (int): Output size for FC Layer
            initializer (Any): Initializer function for FC layer weights
            activation_fn (Any): Activation function at the end of layer
            use_bias (bool): Whether to add bias weights or not
            bias_init (float): Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        layers = []
        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_size))

        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)


class SlimConv1d(nn.Module):
    """Simple mock of tf.slim Conv2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        padding: int,
        # Defaulting these to nn.[..] will break soft torch import.
        initializer: Any = "default",
        activation_fn: Any = "default",
        bias_init: float = 0,
    ):
        """Creates a standard Conv2d layer, similar to torch.nn.Conv2d

        Args:
            in_channels(int): Number of input channels
            out_channels (int): Number of output channels
            kernel (Union[int, Tuple[int, int]]): If int, the kernel is
                a tuple(x,x). Elsewise, the tuple can be specified
            stride (Union[int, Tuple[int, int]]): Controls the stride
                for the cross-correlation. If int, the stride is a
                tuple(x,x). Elsewise, the tuple can be specified
            padding (Union[int, Tuple[int, int]]): Controls the amount
                of implicit zero-paddings during the conv operation
            initializer (Any): Initializer function for kernel weights
            activation_fn (Any): Activation function at the end of layer
            bias_init (float): Initalize bias weights to bias_init const
        """
        super(SlimConv1d, self).__init__()
        layers = []
        # Padding layer.
        if padding:
            layers.append(nn.ZeroPad2d(padding))
        # Actual Conv2D layer (including correct initialization logic).
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)
        # Activation function (if any; default=ReLu).
        if isinstance(activation_fn, str):
            if activation_fn == "default":
                activation_fn = nn.ReLU
            else:
                activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)
