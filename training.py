from copy import deepcopy
from dataclasses import dataclass
from operator import itemgetter
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from data_utils import DatasetInfo
from regularization import ActivationRegularizer
from typing import Any, Dict, Iterable, List, Tuple

from data_utils import repeat


@dataclass
class OptimParams:
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    epochs: int
    reg_factor: float
    regularizer: callable = ActivationRegularizer
    reg_period: int = 4
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DataParams:
    info: DatasetInfo
    train_loader: DataLoader
    test_loader: DataLoader
    val_loader: DataLoader | None = None
    repeats: int = 1
    recursions: int = 1


@dataclass
class TrainParams:
    optim: OptimParams
    data: DataParams


@dataclass
class EvalResult:
    report: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


class RunningMetrics(defaultdict):
    """
    A defaultdict subclass for accumulating and averaging metrics.
    """

    def __init__(self):
        super().__init__(float)
        self.count: int = 0

    def reset(self):
        self.count = 0

    def update(self, metrics: Dict[str, float]):
        """
        Updates the accumulated metrics with new values.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
        """
        for k, v in metrics.items():
            self[k] += v
        self.count += 1

    def _get_as(self, prefix: str = "") -> Dict[str, float]:
        """
        Returns the averaged metrics with an optional prefix.

        Args:
            prefix (str, optional): A prefix to add to the metric names. Defaults to "".

        Returns:
            Dict[str, float]: A dictionary of averaged metrics.
        """
        prefix = f"{prefix}_" if len(prefix) else ""
        return {(prefix + k): v / self.count for k, v in self.items()}


class TrainingOperation:
    """
    Base class for training operations.
    """

    def __init__(self, params: TrainParams, logger: object | None = None):
        self.params = params
        self.logger = logger

    def __call__(self, *args, **kwargs):
        """
        Calls the training operation with the given task.
        """
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs) -> Any:
        """
        Runs the training operation with the given task.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _format_data(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Formats the input and label tensors for the model.
        """
        inputs, labels = inputs.to(self.optim.device), labels.to(self.optim.device)
        if not self.data.info.is_timeseries:
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
        return inputs, labels

    def _get_loader(self, section: str) -> DataLoader:
        loader = None
        if section == "train":
            loader = self.data.train_loader
        elif section == "test":
            loader = self.data.test_loader
        elif section == "val":
            loader = self.data.val_loader

        return loader

    def __getattr__(self, name):
        return getattr(self.params, name)


class Trainer(TrainingOperation):
    """
    Trains a neural network model.
    """

    def _iter(
        self, inputs: torch.Tensor, labels: torch.Tensor, epoch: int
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs a single training pass for a batch.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor): The labels tensor.
            epoch (int): The current epoch number.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - The calculated loss.
                - A tuple containing main_loss, reg_term, and factor.
        """
        output, out_of_dist = self.optim.model(inputs)

        factor = out_of_dist.abs().detach()
        main_loss = (factor * self.optim.criterion(output, labels)).mean()

        regs = self.reg_handler.get()

        if self._cosine is None:

            self._cosine = (
                torch.tensor(2 * torch.pi * epoch / self.optim.reg_period)
                .cos()
                .add(2)
                .div(4)
            )

        reg_term = (
            torch.lerp(regs["energy"], regs["flow"], self._cosine)
            .add(regs["sparsity"])
            .mul(factor)
            .mean()
            .add(regs["quant"])
        )
        factor_term = out_of_dist.mean() ** 2

        loss = main_loss + self.optim.reg_factor * (reg_term + factor_term)

        return loss, (main_loss, reg_term, factor)

    def run(self) -> nn.Module | None:
        """
        Runs the training loop.

        Returns:
            Optional[nn.Module]: The best model based on validation score, or None if no validation is performed.
        """
        model = self.optim.model.to(self.optim.device)
        criterion = self.optim.criterion.to(self.optim.device)
        optimizer = self.optim.optimizer

        evaluator = Evaluator(self.params, logger=self.logger)
        self.reg_handler = self.optim.regularizer(model)

        repeats = self.data.repeats
        recursions = self.data.recursions

        self.running_score = None
        self.best_model = None

        metrics = RunningMetrics()
        model.train()
        for epoch in range(self.optim.epochs):
            metrics.reset()
            self._cosine = None
            loader = self._get_loader("train")
            pbar = tqdm(
                repeat(loader, repeats),
                total=len(loader) * repeats,
            )
            with self.reg_handler:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._format_data(inputs, labels)

                    optimizer.zero_grad()

                    t_total = inputs.shape[1]
                    for t_step in range(t_total):
                        inputs_step = inputs[:, t_step]
                        labels_step = labels[:, t_step]

                        for r_step in range(recursions):
                            loss, (main_loss_step, reg_term_step, factor_step) = (
                                self._iter(inputs_step, labels_step, epoch)
                            )
                            loss.backward()

                            denom = t_total * recursions
                            metrics.update(
                                {
                                    "loss": main_loss_step.item() / denom,
                                    "reg": reg_term_step.item() / denom,
                                    "factor": factor_step.mean().item() / denom,
                                }
                            )

                        optimizer.step()
                        model.reset_state()

                    pbar.set_description(
                        f"Epoch {epoch+1}/{self.optim.epochs} - @ {metrics['factor'] / (d:=i+1):.4f} - Loss: {metrics['loss'] / d:.4f} - Reg: {metrics['reg'] / d:.4f}"
                    )

            if self.logger:
                self.logger.log_metrics(
                    {k: v for k, v in metrics._get_as("train").items()}, epoch
                )

            model.eval()
            if loader_ := self._get_loader("val"):
                val_metrics = RunningMetrics()
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(loader_):
                        inputs, labels = self._format_data(inputs, labels)

                        model.reset_state()
                        t_total = inputs.shape[1]
                        for t_step in range(t_total):
                            inputs_step = inputs[:, t_step]
                            labels_step = labels[:, t_step]
                            output, _ = model(inputs_step)
                            loss = criterion(output, labels_step).mean()
                            val_metrics.update({"loss": loss.item() / t_total})

                evaluation = evaluator("val")
                if self.data.info.is_classify:
                    acc, auc, f1 = itemgetter("acc", "auc", "f1")(evaluation.report)
                    val_score = auc + (f1**0.5).mean()

                    print(
                        f"* Val Loss: {val_metrics['loss']:.4f} - Acc: {acc:.4f} - AUC: {auc:.4f} - F1: {f1}"
                    )
                else:
                    mae, mse = itemgetter("mae", "mse")(evaluation.report)
                    val_score = -(mae + mse**0.5)

                    print(
                        f"* Val Loss: {val_metrics['loss']:.4f} - MAE: {mae:.4f} - MSE: {mse:.4f}"
                    )

                if self.logger:
                    self.logger.log_metrics(
                        dict(
                            **{
                                f"val_{k}_{i}": t
                                for k, v in evaluation.report.items()
                                if isinstance(v, Iterable)
                                for i, t in enumerate(v)
                            },
                            val_loss=val_metrics["loss"],
                        ),
                        epoch,
                    )

                if self.best_model is None:
                    self.running_score = val_score

                ratio = (0.99, 0.01)
                if val_score >= self.running_score:
                    self.best_model = deepcopy(model)
                    print("-- New best model --")
                    ratio = (0.7, 0.3)

                self.running_score = (
                    self.running_score * ratio[0] + val_score * ratio[1]
                )

        return self.best_model


class Evaluator(TrainingOperation):
    """
    Evaluates a neural network model on a test dataset.
    """

    def run(self, section: str = "test") -> EvalResult:
        """
        Evaluates the network on the test dataset.

        Returns:
            EvalResult: An object containing the evaluation metrics, report, and predictions.

        """
        y_true: List[np.ndarray] = []
        y_pred: List[np.ndarray] = []
        y_proba: List[np.ndarray] = []

        loader = self._get_loader(section)
        if loader is None:
            raise ValueError("The section must be 'train', 'val', or 'test'.")

        model = self.optim.model.to(self.optim.device)
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = self._format_data(inputs, labels)

                model.reset_state()
                for t_step in range(inputs.shape[1]):
                    inputs_step = inputs[:, t_step]
                    labels_step = labels[:, t_step]
                    output, _ = model(inputs_step)

                    if not self.data.info.is_classify:
                        predicted = output.cpu()
                        proba = predicted
                    else:
                        proba = torch.softmax(output, dim=1).cpu()
                        predicted = torch.argmax(proba, dim=1).cpu()

                    y_true.extend(labels_step.cpu().numpy())
                    y_pred.extend(predicted.numpy())
                    y_proba.extend(proba.numpy())

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        y_proba_np = np.array(y_proba)

        if self.data.info.is_classify:
            acc = accuracy_score(y_true_np, y_pred_np)
            if isinstance((n_ := self.data.info.n_targets), int) and n_ == 2:
                auc = roc_auc_score(
                    y_true_np, [p for _, p in y_proba_np], average="macro"
                )
            else:
                auc = roc_auc_score(
                    y_true_np, y_proba_np, multi_class="ovr", average="macro"
                )

            kwargs = {"average": None, "zero_division": 0}

            prec = precision_score(y_true_np, y_pred_np, **kwargs)
            rec = recall_score(y_true_np, y_pred_np, **kwargs)
            f1 = f1_score(y_true_np, y_pred_np, **kwargs)

            report = {
                "acc": acc,
                "auc": auc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        else:
            mae = np.abs(np.array(y_true_np) - np.array(y_pred_np)).mean()
            mse = np.square(np.array(y_true_np) - np.array(y_pred_np)).mean()
            report = {
                "mae": mae,
                "mse": mse,
            }

        return EvalResult(
            report=report,
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_proba=y_proba_np,
        )
