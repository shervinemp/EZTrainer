from typing import Any, Callable
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from functools import partial


class PaddedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, kernel_size=3, padding="same")


class AdaptiveBatchNorm(nn.Module):

    def __init__(
        self,
        input_dim: int,
        *,
        alpha: float = 0.95,
        eps: float = 0.01,
        jitter: float = 1e-4,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.alpha = alpha
        self.eps = eps
        self.jitter = jitter
        self.dtype = dtype

        self.weight: nn.Parameter = nn.Parameter(
            torch.ones((1, input_dim), dtype=dtype)
        )
        self.bias: nn.Parameter = nn.Parameter(torch.zeros((1, input_dim), dtype=dtype))

        self.register_buffer("mean_ema", None)
        self.register_buffer("std_ema", None)

        self.register_buffer(
            "_count", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_cum_mean", torch.zeros((1, input_dim), dtype=dtype), persistent=False
        )
        self.register_buffer(
            "_cum_mean_sq", torch.zeros((1, input_dim), dtype=dtype), persistent=False
        )

        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra_dims = (1,) * (x.dim() - 2)
        mean = self.mean_ema.view(1, -1, *extra_dims)
        std = self.std_ema.view(1, -1, *extra_dims)
        x_norm = (x - mean) / (std + self.eps)

        view_shape = self.weight.shape + extra_dims
        r = self.training * torch.rand(view_shape, device=x.device) * self.jitter
        w = self.weight.view(view_shape) * r.exp()
        b = self.bias.view(view_shape) - r
        o = x_norm * w + b

        return o

    def _backward_hook(self, module: nn.Module, grad_input: Any, grad_output: Any):
        self._count.zero_()

    @torch.no_grad()
    def _forward_pre_hook(self, module: nn.Module, args: tuple):
        if not self.training:
            return

        x = args[0]

        if x.dim() > 2:
            x_flat = x.transpose(1, -1).reshape(-1, self.input_dim)
        else:
            x_flat = x

        old_count = self._count.clone()
        old_mean = self.mean_ema
        old_std = self.std_ema

        self._count += x_flat.shape[0]

        if old_count == 0:
            if old_mean is None:
                self.mean_ema = torch.mean(x_flat, dim=0, keepdim=True).detach()
                self.std_ema = torch.std(x_flat, dim=0, keepdim=True).detach()
            else:
                c_std = torch.clamp(self._cum_mean_sq - self._cum_mean**2, 0.0) ** 0.5
                factor = 1 - self.alpha
                self.mean_ema.data.add_((self._cum_mean - old_mean) * factor)
                self.std_ema.data.add_((c_std - old_std) * factor)
                self._adjust_weight(old_mean, old_std, self.mean_ema, self.std_ema)

            self._cum_mean.zero_()
            self._cum_mean_sq.zero_()

        self._cum_mean += (x_flat - self._cum_mean).sum(dim=0, keepdim=True) / self._count
        self._cum_mean_sq += (x_flat**2 - self._cum_mean_sq).sum(
            dim=0, keepdim=True
        ) / self._count

    @torch.no_grad()
    def _adjust_weight(self, old_mean, old_std, new_mean, new_std):
        """
        Adjusts the weight and bias based on the new mean and std.
        """
        new_std = new_std + self.eps
        old_std = old_std + self.eps

        scale = new_std / old_std
        shift = new_mean - old_mean

        self.weight.data.mul_(scale)
        self.bias.data.div_(scale)
        self.bias.data.add_(shift / new_std)


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

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_module: nn.Module = nn.Linear,
        bias: bool = True,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_module = inner_module

        self.layers = nn.ModuleDict(
            {
                "affine": AdaptiveBatchNorm(input_dim, dtype=dtype),
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
        dropout (float, optional): Dropout probability. 0 disables dropout.
            Defaults to 0.0.
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
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_module = inner_module

        layers = {
            "affine": AffineBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                inner_module=inner_module,
                bias=bias,
                dtype=dtype,
            ),
            "skip": SkipBlock(
                input_dim=output_dim,
                activation=activation,
                activation_params=activation_params or {},
            ),
        }
        if dropout > 0:
            layers["dropout"] = nn.Dropout(dropout)
        self.layers = nn.ModuleDict(layers)

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
        if "dropout" in self.layers:
            x = self.layers["dropout"](x)
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
        *,
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
                "query_key": inner_module(
                    input_dim,
                    (heads + 1) * output_dim,
                    bias=bias,
                ),
                "value": inner_module(
                    input_dim,
                    output_dim,
                    bias=bias,
                ),
            }
        )

    def reset_state(self):
        """Resets the state of the inner blocks."""
        if hasattr(self.layers["query_key"], "reset_state"):
            self.layers["query_key"].reset_state()
        if hasattr(self.layers["value"], "reset_state"):
            self.layers["value"].reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, *spatial_dims = x.shape

        d, h = self.output_dim, self.heads

        query, key = self.layers["query_key"](x).split([h * d, d], dim=1)
        value = self.layers["value"](x)

        query = query.view(b, h, d, -1)
        key = key.view(b, d, -1)
        value = value.view(b, d, -1)

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
        dropout: float = 0.0,
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
        self.dropout: float = dropout
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
            dropout=getattr(self, "dropout", 0.0),
            dtype=self.dtype,
        )
        atten_ = partial(
            AttentionBlock,
            inner_module=self.inner_module,
            bias=self.bias,
        )
        self.layers = nn.ModuleDict(
            {
                "forward_": nn.ModuleList(
                    [
                        inner_(i_dim * 2, o_dim),
                        *(inner_(o_dim, o_dim) for _ in range(r_dim - 1)),
                    ]
                ),
                "backward_": atten_(r_dim * s_dim, i_dim),
            }
        )

        self.register_buffer("prev_state_", None, persistent=False)
        self.register_buffer("cur_state_", None, persistent=False)
        self.register_forward_pre_hook(self._forward_pre_hook)

    def reset_state(self):
        self.cur_state_ = None
        self.prev_state_ = None

        for layer in self.layers["forward_"]:
            if hasattr(layer, "reset_state"):
                layer.reset_state()
        layer = self.layers["backward_"]
        if hasattr(layer, "reset_state"):
            layer.reset_state()

    def _forward_pre_hook(self, module: nn.Module, args: tuple):
        """
        This hook performs the forward pre-processing.
        """
        x = args[0]
        r_dim = self.recurrent_dim
        s_dim = self.state_dim

        prev_state = self.cur_state_
        if prev_state is None:
            prev_state = torch.zeros(
                (x.shape[0], s_dim * r_dim, *x.shape[2:]),
                dtype=x.dtype, device=x.device
            )

        self.prev_state_ = prev_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the recurrent block transformations.
        """
        s_dim = self.state_dim

        h = self.layers["backward_"](self.prev_state_)
        x = torch.cat([x, h], dim=1)

        state_parts = []
        for i, layer in enumerate(self.layers["forward_"]):
            x = layer(x)
            state_parts.append(x[:, :s_dim])

        self.cur_state_ = torch.cat(state_parts, dim=1)
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
        dropout (float, optional): Dropout probability. Defaults to 0.0.
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
        dropout: float = 0.0,
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
        self.dropout = dropout
        self.collapse_output = collapse_output
        self.dtype = dtype

        inner_block = partial(
            inner_block, inner_module=inner_module, dropout=dropout, dtype=dtype
        )
        layers = {
            "input": inner_block(input_dim, hidden_dim),
            "hidden": nn.ModuleList(
                [inner_block(hidden_dim, hidden_dim) for _ in range(n_hidden)]
            ),
            "output": nn.ModuleList(
                [
                    inner_module(hidden_dim, task_dim, dtype=dtype)
                    for task_dim in output_dim
                ]
            ),
            "log_var": inner_module(hidden_dim, 1, bias=False, dtype=dtype),
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

        if torch.is_complex(o):
            o = o.real

        if self.collapse_output and (dims := tuple(range(2, len(o.shape)))):
            o = o.mean(dim=dims)

        log_var = self.layers["log_var"](x)
        if torch.is_complex(log_var):
            log_var = log_var.real
        if self.collapse_output and (dims := tuple(range(2, len(log_var.shape)))):
            log_var = log_var.mean(dim=dims)

        return o, log_var
