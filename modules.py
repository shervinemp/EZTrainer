from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from functools import partial


class PaddedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kernel_size=3, padding="same")


class Affine(nn.Module):
    """
    Applies a learnable affine transformation (scaling and shift) to the input tensor.

    This module performs an element-wise scaling and shifting of the input,
    controlled by learnable parameters `weight` and `bias`. It can optionally
    apply jitter during training.

    Args:
        input_dim (int): The dimension of the input feature space.
        jitter (float, optional): The standard deviation of the random noise
            applied to weight and bias during training. Defaults to 0.005.
        dtype (torch.dtype, optional): The data type of the module parameters.
            Defaults to torch.float32.

    Attributes:
        weight (nn.Parameter): Learnable scaling parameter.
        bias (nn.Parameter): Learnable shifting parameter.
        jitter (torch.Tensor): Jitter value as a tensor.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        jitter: float = 0.005,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.weight: nn.Parameter = nn.Parameter(
            torch.ones((1, input_dim), dtype=dtype)
        )
        self.bias: nn.Parameter = nn.Parameter(torch.zeros((1, input_dim), dtype=dtype))
        self.jitter: torch.Tensor = torch.tensor(jitter, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the affine transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the affine transformation.
        """
        view_shape = self.weight.shape + (1,) * (len(x.shape) - 2)
        w = self.weight.view(view_shape)
        b = self.bias.view(view_shape)
        if self.training and self.jitter:
            w = w * torch.exp2(torch.randn_like(x) * self.jitter)
            b = b + torch.randn_like(w) * self.jitter
        return (x + b) * w


class QuickSkip(nn.Module):
    """
    Implements a quick skip connection with learnable transformation and carry gates.

    This module takes two inputs, `z` and `x`, and combines them using a learnable
    coefficient to create a skip connection.

    Args:
        input_dim (int): The dimension of the input feature space for both `z` and `x`.

    Attributes:
        input_dim (int): The dimension of the input feature space.
        _coeff (nn.Parameter): Learnable coefficient controlling the gates.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self._coeff: nn.Parameter = nn.Parameter(
            torch.normal(mean=2 / 3, std=0.1, size=(1,))
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the quick skip connection.

        Args:
            z (torch.Tensor): The tensor to be transformed.
            x (torch.Tensor): The tensor to be carried.

        Returns:
            torch.Tensor: The output tensor after applying the skip connection.
        """
        transform_gate = torch.abs(self._coeff)
        carry_gate = torch.abs(1 - self._coeff**2) * 2
        output = transform_gate * z + carry_gate * x
        return output


class BaseBlock(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_state(self):
        """Resets the internal state of the block, if any."""
        pass


class AffineBlock(BaseBlock):
    """
    A block combining an Affine transformation and an inner module (e.g., Linear).

    Args:
        input_dim (int): The dimension of the input feature space.
        output_dim (int): The dimension of the output feature space.
        inner_module (nn.Module, optional): The inner module to apply after the
            Affine transformation. Defaults to nn.Linear.
        bias (bool, optional): Whether the inner module should include a bias term.
            Defaults to True.
        jitter (float, optional): Jitter parameter for the Affine layer.
            Defaults to 0.005.
        dtype (torch.dtype, optional): Data type for the layers.
            Defaults to torch.float32.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_module: nn.Module = nn.Linear,
        bias: bool = True,
        *,
        jitter: float = 0.005,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_module = inner_module

        self.layers = nn.ModuleDict(
            {
                "affine": Affine(input_dim, jitter=jitter, dtype=dtype),
                "inner": inner_module(input_dim, output_dim, bias=bias, dtype=dtype),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Affine and inner module transformations.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.layers["affine"](x)
        x = self.layers["inner"](x)
        return x

    def reset_state(self):
        """Resets the state of the inner layers if they have a reset_state method."""
        for layer in self.layers.values():
            if hasattr(layer, "reset_state"):
                layer.reset_state()


class SkipBlock(BaseBlock):
    """
    A block implementing a skip connection with an activation function and QuickSkip.

    Args:
        input_dim (int): The dimension of the input feature space.
        activation (nn.Module, optional): The activation function to apply before
            the QuickSkip connection. Defaults to torch.tanh.
        activation_params (dict, optional): Additional parameters for the activation
            function. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        activation: nn.Module | Callable = torch.tanh,
        *,
        activation_params: dict = None,
    ):
        super().__init__()
        self.input_dim: int = input_dim
        self.activation: nn.Module | Callable = activation
        self.activation_params: dict[str, float] = activation_params or {}

        self.layers = nn.ModuleDict(
            {
                "skip": QuickSkip(self.input_dim),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the activation and skip connection.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        z = self.activation(x, **self.activation_params)
        x = self.layers["skip"](z, x)
        return x

    def reset_state(self):
        """Resets the state of the inner layers if they have a reset_state method."""
        for layer in self.layers.values():
            if hasattr(layer, "reset_state"):
                layer.reset_state()


class UnitBlock(BaseBlock):
    """
    A block combining an AffineBlock and a SkipBlock.

    Args:
        input_dim (int): The dimension of the input feature space.
        output_dim (int): The dimension of the output feature space.
        inner_module (nn.Module, optional): The inner module for the AffineBlock.
            Defaults to nn.Linear.
        bias (bool, optional): Whether the inner module should include a bias term.
            Defaults to True.
        activation (nn.Module, optional): The activation function for the SkipBlock.
            Defaults to torch.tanh.
        activation_params (dict, optional): Additional parameters for the activation
            function. Defaults to None.
        jitter (float, optional): Jitter parameter for the AffineBlock.
            Defaults to 0.005.
        dtype (torch.dtype, optional): Data type for the layers.
            Defaults to torch.float32.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_module: nn.Module = nn.Linear,
        bias: bool = True,
        activation: nn.Module | Callable = torch.tanh,
        *,
        activation_params: dict = None,
        jitter: float = 0.005,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_module = inner_module

        self.layers = nn.ModuleDict(
            {
                "affine": AffineBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    inner_module=inner_module,
                    bias=bias,
                    jitter=jitter,
                    dtype=dtype,
                ),
                "skip": SkipBlock(
                    input_dim=output_dim,
                    activation=activation,
                    activation_params=activation_params or {},
                ),
            }
        )

    def reset_state(self):
        """Resets the state of the inner blocks."""
        self.layers["affine"].reset_state()
        self.layers["skip"].reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the AffineBlock and SkipBlock transformations.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.layers["affine"](x)
        x = self.layers["skip"](x)

        return x


class AttentionBlock(BaseBlock):
    """
    A multi-head linear attention block.

    Args:
        input_dim (int): The dimension of the input feature space.
        output_dim (int): The dimension of the output feature space.
        inner_module (nn.Module, optional): The inner module for the AffineBlock.
            Defaults to nn.Linear.
        bias (bool, optional): Whether the inner module should include a bias term.
            Defaults to True.
        activation (nn.Module, optional): The activation function for the SkipBlock.
            Defaults to torch.tanh.
        kernel_activation (nn.Module, optional): The kernel function phi for attention.
            Defaults to F.elu(x) + 1.
        activation_params (dict, optional): Additional parameters for the activation
            function. Defaults to None.
        jitter (float, optional): Jitter parameter for the AffineBlock.
            Defaults to 0.005.
        dtype (torch.dtype, optional): Data type for the layers.
            Defaults to torch.float32.
        eps (float, optional): Epsilon for numerical stability in normalization.
            Defaults to 1e-6.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int = 1,
        inner_module: nn.Module = nn.Linear,
        bias: bool = True,
        activation: nn.Module | Callable = torch.tanh,
        *,
        kernel_activation: nn.Module | Callable = lambda x: F.elu(x) + 1,
        activation_params: dict = None,
        jitter: float = 0.005,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.inner_module = inner_module
        self.eps: float = eps

        self.layers = nn.ModuleDict(
            {
                "query": UnitBlock(
                    input_dim=input_dim,
                    output_dim=heads * output_dim,
                    inner_module=inner_module,
                    bias=bias,
                    activation=kernel_activation,
                    jitter=jitter,
                    dtype=dtype,
                ),
                "key": UnitBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    inner_module=inner_module,
                    bias=bias,
                    activation=kernel_activation,
                    jitter=jitter,
                    dtype=dtype,
                ),
                "value": UnitBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    inner_module=inner_module,
                    bias=bias,
                    activation=activation,
                    activation_params=activation_params or {},
                    jitter=jitter,
                    dtype=dtype,
                ),
            }
        )

    def reset_state(self):
        """Resets the state of the inner blocks."""
        self.layers["query"].reset_state()
        self.layers["key"].reset_state()
        self.layers["value"].reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, *spatial_dims = x.shape

        d, h = self.output_dim, self.heads

        query = self.layers["query"](x).view(b, h, d, -1)
        key = self.layers["key"](x).view(b, d, -1)
        value = self.layers["value"](x).view(b, d, -1)

        k_sum = key.sum(dim=2)
        context = torch.einsum("bdn,ben->bde", key, value)

        D = torch.einsum("bhdn,bd->bhn", query, k_sum)
        out = torch.einsum("bhdn,bde->bhen", query, context)

        normalized_out = out / D.unsqueeze(2).clamp(min=self.eps)

        return normalized_out.view(b, h * d, *spatial_dims)


class RecurrentBlock(BaseBlock):
    """
    A recurrent block with forward and backward connections.

    Args:
        input_dim (int): The dimension of the input feature space.
        output_dim (int): The dimension of the output feature space.
        recurrent_dim (int, optional): The dimension of the recurrent state.
            Defaults to 2.
        inner_module (nn.Module, optional): The inner module for the UnitBlocks.
            Defaults to nn.Linear.
        bias (bool, optional): Whether the inner module should include a bias term.
            Defaults to True.
        activation (nn.Module, optional): The activation function for the UnitBlocks.
            Defaults to torch.tanh.
        activation_params (dict, optional): Additional parameters for the activation
            function. Defaults to None.
        jitter (float, optional): Jitter parameter for the UnitBlocks.
            Defaults to 0.005.
        dtype (torch.dtype, optional): Data type for the layers.
            Defaults to torch.float32.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        recurrent_dim: int = 2,
        inner_module: nn.Module = nn.Linear,
        bias: bool = True,
        activation: nn.Module | Callable = torch.tanh,
        *,
        activation_params: dict = None,
        jitter: float = 0.005,
        dtype: torch.dtype = torch.float32,
    ):

        super().__init__()
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.recurrent_dim: int = recurrent_dim
        self.inner_module: nn.Module = inner_module
        self.bias: bool = bias
        self.activation: nn.Module | Callable = activation
        self.activation_params: dict[str, float] = activation_params or {}
        self.jitter: float = jitter
        self.dtype: torch.dtype = dtype
        self.state_dim = (output_dim + 1) // 2

        i_dim = self.input_dim
        o_dim = self.output_dim
        r_dim = self.recurrent_dim
        s_dim = self.state_dim

        inner_ = partial(
            UnitBlock,
            inner_module=self.inner_module,
            bias=self.bias,
            activation=self.activation,
            activation_params=self.activation_params,
            jitter=self.jitter,
            dtype=self.dtype,
        )
        atten_ = partial(
            AttentionBlock,
            inner_module=self.inner_module,
            bias=self.bias,
            activation=self.activation,
            activation_params=self.activation_params,
            jitter=self.jitter,
            dtype=self.dtype,
        )
        self.layers = nn.ModuleDict(
            {
                "forward_": nn.ModuleList(
                    [
                        atten_(i_dim * 2, o_dim),
                        *(inner_(o_dim, o_dim) for _ in range(r_dim - 1)),
                    ]
                ),
                "backward_": inner_(r_dim * s_dim, i_dim),
            }
        )

        prev_state_ = torch.zeros((1, r_dim * s_dim), dtype=torch.float32)
        self.register_buffer("prev_state_", prev_state_)
        self.register_full_backward_hook(self._detach_prev_hook)

    def reset_state(self):
        """Resets the previous state of the recurrent block."""
        self.prev_state_ = 0 * self.prev_state_

    def _detach_prev_hook(self, module, grad_input, grad_output):
        self.prev_state_ = self.prev_state_.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the recurrent block transformations.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        r_dim = self.recurrent_dim
        s_dim = self.state_dim

        if self.prev_state_.shape != (
            new_shape := (x.shape[0], s_dim * r_dim, *x.shape[2:])
        ):
            self.prev_state_ = torch.broadcast_to(self.prev_state_, new_shape).clone()

        h = self.layers["backward_"](self.prev_state_)
        x = torch.cat([x, h], dim=1)

        for i, layer in enumerate(self.layers["forward_"]):
            x = layer(x)
            self.prev_state_[:, i * s_dim : (i + 1) * s_dim] = x[:, :s_dim]

        return x


class Network(nn.Module):
    """
    The main neural network model composed of various blocks.

    Args:
        input_dim (int): The dimension of the input feature space.
        hidden_dim (int): The dimension of the hidden layers.
        n_hidden (int): The number of hidden layers.
        output_dim (int | tuple[int]): The dimension(s) of the output layer(s).
        inner_module (nn.Module, optional): The inner module to use within blocks.
            Defaults to nn.Linear.
        inner_block (BaseBlock, optional): The type of block to use for hidden layers.
            Defaults to UnitBlock.
        activation (nn.Module, optional): The activation function to use within blocks.
            Defaults to torch.tanh.
        activation_params (dict, optional): Additional parameters for the activation
            function. Defaults to None.
        collapse_output (bool, optional): Whether to collapse the output dimensions.
            Defaults to True.
        dtype (torch.dtype, optional): Data type for the network parameters.
            Defaults to torch.float32.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_hidden: int,
        output_dim: int | tuple[int],
        inner_module: nn.Module = nn.Linear,
        inner_block: BaseBlock = UnitBlock,
        activation: nn.Module | Callable = torch.tanh,
        *,
        activation_params: dict = None,
        collapse_output: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super(Network, self).__init__()

        if isinstance(output_dim, int):
            output_dim = (output_dim,)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.inner_module = inner_module
        self.inner_block = inner_block
        self.activation = activation
        self.activation_params = activation_params or {}
        self.collapse_output = collapse_output
        self.dtype = dtype

        inner_block = partial(inner_block, inner_module=inner_module, dtype=dtype)
        layers = {
            "input": inner_block(input_dim, hidden_dim),
            "hidden": nn.ModuleList(
                [inner_block(hidden_dim, hidden_dim) for _ in range(n_hidden)]
            ),
            "output": nn.ModuleList(
                [
                    inner_module(hidden_dim, task_dim, bias=False, dtype=dtype)
                    for task_dim in output_dim
                ]
            ),
            "distrib": inner_module(hidden_dim, 1, dtype=dtype),
        }

        self.layers = nn.ModuleDict(layers)

    def reset_state(self):
        self.layers["input"].reset_state()
        for layer in self.layers["hidden"]:
            layer.reset_state()

    def forward(self, x):
        x = x.to(self.dtype)
        x = self.layers["input"](x)

        for layer in self.layers["hidden"]:
            x = layer(x)

        outputs = []
        for layer in self.layers["output"]:
            o = layer(x)
            outputs.append(o)

        o = torch.cat(outputs, dim=1)
        d = self.layers["distrib"](x)

        if torch.is_complex(o):
            o = o.real
            d = d.real

        if self.collapse_output and (dims := tuple(range(2, len(o.shape)))):
            o = o.mean(dim=dims)
            d = d.mean(dim=dims)

        return o, d
