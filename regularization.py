import torch
import torch.nn as nn


class ActivationRegularizer:
    __slots__ = (
        "model",
        "module_type",
        "quantize",
        "count",
        "hooks",
        "energy_term",
        "flow_term",
        "sparsity_term",
        "quant_term",
    )

    def __init__(
        self,
        model: nn.Module,
        module_type: nn.Module = nn.Linear,
        quantize: bool = False,
    ):
        self.model: nn.Module = model
        self.module_type = module_type
        self.quantize = quantize
        self._init()

    def _init(self):
        self.count = 0
        self.hooks = []

    def _reset_terms(self, module, input):
        self.sparsity_term = 0
        self.energy_term = 0
        self.flow_term = 0
        self.quant_term = 0

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hooks()

    def _register_hooks(self):
        self._init()
        for name, module in self.model.named_modules():
            if isinstance(module, self.module_type):
                self.count += 1
                hook = module.register_forward_hook(self._hook)
                self.hooks.append(hook)

        self.count = torch.tensor(self.count)
        hook = self.model.register_forward_pre_hook(self._reset_terms)
        self.hooks.append(hook)

    def _hook(self, module, input, output):
        lin_l2_ = output.abs().square().view(output.size(0), -1).mean(dim=1)
        flow_ = module.weight.abs().sum(dim=1).neg().exp().mean()
        wgt_log_ = module.weight.abs().add(1).log2().mean()

        # x_ = torch.pi * (2 * output.sgn() * output.abs().log2() - 1)
        # int_ = self.quantize or (1- (1 - torch.cos(x_)) ** 2 / 4).mean()

        self.energy_term += lin_l2_
        self.flow_term += flow_
        self.sparsity_term += wgt_log_
        # self.quant_term += int_

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get(self):
        return {
            "energy": self.energy_term,
            "flow": self.flow_term,
            "sparsity": self.sparsity_term,
            # "quant": self.quant_term,
        }
