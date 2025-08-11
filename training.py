from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from data_utils import TaskConfig, Task
from regularization import ActivationRegularizer
from typing import Any, List, Optional, Dict, Tuple


@dataclass
class Training:
    model: nn.Module
    task: Task
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    epochs: int
    reg_factor: float
    sparsity_period: int
    regularizer: callable = ActivationRegularizer
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Evaluation:
    metrics: Dict[str, float]
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

    def update(self, metrics: Dict[str, float]):
        """
        Updates the accumulated metrics with new values.

        Args:
            metrics (Dict[str, float]): A dictionary of metric names and their values.
        """
        for k, v in metrics.items():
            self[k] += v
        self.count += 1

    def get_as(self, prefix: str = "") -> Dict[str, float]:
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

    def __init__(self, training: Training, logger: Optional[object] = None):
        self.training = training
        self.logger = logger

    def __call__(self):
        """
        Calls the training operation with the given task.
        """
        return self.run()

    def run(self) -> Any:
        """
        Runs the training operation with the given task.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _format_data(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_config: TaskConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Formats the input and label tensors for the model.
        """
        inputs, labels = inputs.to(self._tr.device), labels.to(self._tr.device)
        if not task_config.timeseries:
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
        return inputs, labels

    @property
    def _tr(self):
        return self.training


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
        output, out_of_dist = self._tr.model(inputs)

        factor = out_of_dist.detach().abs().square().add(1).log2()
        main_loss = (factor * self._tr.criterion(output, labels)).mean()

        regs = self.reg_handler.get()

        if self._cosine is None:
            self._cosine = (
                torch.tensor(2 * epoch / self._tr.sparsity_period * torch.pi)
                .cos()
                .add(1)
                .div(2)
            )

        reg_term = (
            torch.lerp(regs["sparsity"], regs["energy"], self._cosine)
            .add(regs["flow"])
            .add(epoch * regs["quant"])
            .mul(factor)
            .mean()
        )

        flow_term = sum(
            [
                task_.weight.abs().mean(dim=1).square().mean()
                for task_ in self._tr.model.layers["output"]
            ]
        ) / len(self._tr.model.layers["output"])

        b_ = self._tr.model.layers["distrib"].bias
        growth_term = (out_of_dist - b_).square().neg().exp().mean()

        loss = main_loss + self._tr.reg_factor * (reg_term + growth_term + flow_term)
        loss_sq = loss**2 / 2 + loss

        return loss_sq, (main_loss, reg_term, factor)

    def run(self) -> Optional[nn.Module]:
        """
        Runs the training loop.

        Returns:
            Optional[nn.Module]: The best model based on validation score, or None if no validation is performed.
        """
        task = self._tr.task
        config = task.config

        self.reg_handler = self._tr.regularizer(self._tr.model)
        self.evaluator = Evaluator(self._tr, logger=self.logger)

        self.running_score = None
        self.best_model = None

        model = self._tr.model.to(self._tr.device)
        _ = self._tr.criterion.to(self._tr.device)

        model.train()
        for epoch in range(self._tr.epochs):
            self._cosine = None
            metrics = RunningMetrics()
            pbar = tqdm(task.train_loader, total=len(task.train_loader))
            with self.reg_handler:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._format_data(inputs, labels, config)

                    self._tr.optimizer.zero_grad()
                    model.reset_state()

                    t_total = inputs.shape[1]
                    for t_step in range(t_total):
                        inputs_step = inputs[:, t_step]
                        labels_step = labels[:, t_step]

                        loss, (main_loss_step, reg_term_step, factor_step) = self._iter(
                            inputs_step, labels_step, epoch
                        )
                        loss.backward(retain_graph=t_step < t_total - 1)
                        self._tr.optimizer.step()

                        metrics.update(
                            {
                                "loss": main_loss_step.item() / t_total,
                                "reg": reg_term_step.item() / t_total,
                                "factor": factor_step.mean().item() / t_total,
                            }
                        )

                    pbar.set_description(
                        f"Epoch {epoch+1}/{self._tr.epochs} - @ {metrics['factor'] / (d:=i+1):.4f} - Loss: {metrics['loss'] / d:.4f} - Reg: {metrics['reg'] / d:.4f}"
                    )

            if self.logger:
                self.logger.log_metrics(
                    {k: v for k, v in metrics.get_as("train").items()}, epoch
                )

            self._tr.model.eval()
            if task.val_loader:
                val_metrics = RunningMetrics()
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(task.val_loader):
                        inputs, labels = self._format_data(inputs, labels, config)

                        self._tr.model.reset_state()
                        t_total = inputs.shape[1]
                        for t_step in range(t_total):
                            inputs_step = inputs[:, t_step]
                            labels_step = labels[:, t_step]
                            output, _ = self._tr.model(inputs_step)
                            loss = self._tr.criterion(output, labels_step).mean()
                            val_metrics.update({"loss": loss.item() / t_total})

                evaluation = self.evaluator()
                if config.classify:
                    acc, auc = evaluation.metrics
                    val_score = acc + auc**0.5

                    print(
                        f"* Val Loss: {val_metrics['loss']:.4f} - Acc: {acc:.4f} - AUC: {auc:.4f}"
                    )
                    if self.logger:
                        self.logger.log_metrics(
                            dict(
                                **{
                                    f"val_{k}_{i}": t
                                    for k, v in evaluation.report.items()
                                    for i, t in enumerate(v)
                                },
                                accuracy=acc,
                                auc=auc,
                                val_loss=val_metrics["loss"],
                            ),
                            epoch,
                        )
                else:
                    mae, mse = evaluation.metrics
                    val_score = -(mae + mse**0.5)

                    print(
                        f"* Val Loss: {val_metrics['loss']:.4f} - MAE: {mae:.4f} - MSE: {mse:.4f}"
                    )
                    if self.logger:
                        self.logger.log_metrics(
                            {"val_loss": val_metrics["loss"], "mae": mae, "mse": mse},
                            epoch,
                        )

                if self.best_model is None:
                    self.running_score = val_score

                ratio = (0.98, 0.02)
                if val_score >= self.running_score:
                    self.best_model = deepcopy(self._tr.model)
                    print("-- New best model --")
                    ratio = (0.8, 0.2)

                self.running_score = (
                    self.running_score * ratio[0] + val_score * ratio[1]
                )

        return self.best_model


class Evaluator(TrainingOperation):
    """
    Evaluates a neural network model on a test dataset.
    """

    def run(self) -> Evaluation:
        """
        Evaluates the network on the test dataset.

        Returns:
            Evaluation: An object containing the evaluation metrics, report, and predictions.

        """
        task = self._tr.task
        config = task.config

        y_true: List[np.ndarray] = []
        y_pred: List[np.ndarray] = []
        y_proba: List[np.ndarray] = []

        self._tr.model.eval()
        with torch.no_grad():
            for inputs, labels in task.test_loader:
                inputs, labels = self._format_data(inputs, labels, config)

                self._tr.model.reset_state()
                for t_step in range(inputs.shape[1]):
                    inputs_step = inputs[:, t_step]
                    labels_step = labels[:, t_step]
                    output, _ = self._tr.model(inputs_step)

                    if not config.classify:
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

        if config.classify:
            acc = accuracy_score(y_true_np, y_pred_np)
            if isinstance(config.n_targets, int) and config.n_targets == 2:
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

            metrics = (acc, auc)
            report = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        else:
            mae = np.abs(np.array(y_true_np) - np.array(y_pred_np)).mean()
            mse = np.square(np.array(y_true_np) - np.array(y_pred_np)).mean()
            metrics = (mae, mse)
            report = {}

        return Evaluation(
            metrics=metrics,
            report=report,
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_proba=y_proba_np,
        )
