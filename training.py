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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from data_utils import DatasetInfo
from typing import Any, Dict, Iterable, List, Tuple

from data_utils import repeat


@dataclass
class OptimParams:
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    epochs: int
    reg_factor: float
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


class FocalLoss(nn.Module):
    # ... (rest of __init__ is unchanged) ...
    def __init__(
        self,
        gamma=2,
        alpha=None,
        reduction="mean",
        task_type="binary",
        num_classes=None,
    ):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if (
            task_type == "multi-class"
            and alpha is not None
            and isinstance(alpha, (list, torch.Tensor))
        ):
            assert (
                num_classes is not None
            ), "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        # ... (forward pass logic unchanged) ...
        if self.task_type == "binary":
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == "multi-class":
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == "multi-label":
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'."
            )

    def binary_focal_loss(self, inputs, targets):
        """Focal loss for binary classification."""
        # Ensure targets are float type as expected by BCEWL
        targets = targets.float()

        # Compute binary cross entropy
        # CHANGE THIS LINE: Use F.binary_cross_entropy_with_logits instead of F.cross_entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = torch.sigmoid(inputs)  # Move sigmoid calculation here for clarity
        p_t = p_t * targets + (1 - p_t) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    # ... (multi_class_focal_loss and multi_label_focal_loss are unchanged) ...
    def multi_class_focal_loss(self, inputs, targets):
        """Focal loss for multi-class classification."""
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)
        targets = targets.long()  # Ensure targets are long for one_hot

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        # Note: Using -log(probs) is numerically less stable than F.cross_entropy directly
        ce_loss = -targets_one_hot * torch.log(probs.clamp(min=1e-6))

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """Focal loss for multi-label classification."""
        # Ensure targets are float type as expected by BCEWL
        targets = targets.float()
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            # Note: Alpha handling might need adjustment if self.alpha is intended to be a scalar/single tensor
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskFocalLoss(nn.Module):
    def __init__(self, task_dims: Tuple[int, ...], gamma=2, reduction="mean"):
        super().__init__()
        self.task_dims = task_dims
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        splits = list(self.task_dims)
        input_splits = torch.split(inputs, splits, dim=-1)
        target_splits = torch.split(targets, splits, dim=-1)

        total_loss = 0
        for o, t in zip(input_splits, target_splits):
            t_class = t.argmax(dim=-1)
            ce_loss = F.cross_entropy(o, t_class, reduction="none")
            probs = F.softmax(o, dim=-1)
            p_t = probs.gather(1, t_class.unsqueeze(-1)).squeeze(-1)
            focal_weight = (1 - p_t) ** self.gamma
            total_loss = total_loss + ce_loss * focal_weight

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        return total_loss


class Trainer(TrainingOperation):
    """
    Trains a neural network model.
    """

    def _iter(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        model = self.optim.model
        output, log_var = model(inputs)

        task_loss = self.optim.criterion(output, labels)
        if task_loss.dim() > 1:
            task_loss = task_loss.mean(dim=tuple(range(1, task_loss.dim())))

        log_var = log_var.squeeze(-1).clamp(min=-10, max=10)
        precision = (-log_var).exp()
        main_loss = (precision * task_loss).mean()
        reg_loss = 0.5 * log_var.mean()
        loss = main_loss + reg_loss

        return loss, (main_loss.detach(), reg_loss.detach(), precision.mean().detach())

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

        repeats = self.data.repeats
        recursions = self.data.recursions

        self.best_score = -float("inf")
        self.best_model = None

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )

        metrics = RunningMetrics()
        model.train()
        for epoch in range(self.optim.epochs):
            metrics.reset()
            loader = self._get_loader("train")
            pbar = tqdm(
                repeat(loader, repeats),
                total=len(loader) * repeats,
            )
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = self._format_data(inputs, labels)

                model.reset_state()
                optimizer.zero_grad()

                t_total = inputs.shape[1]
                for t_step in range(t_total):
                    inputs_step = inputs[:, t_step]
                    labels_step = labels[:, t_step]

                    for r_step in range(recursions):
                        loss, (main_loss_step, reg_loss_step, factor_step) = self._iter(
                            inputs_step, labels_step
                        )
                        loss = loss / (t_total * recursions)
                        loss.backward()
                        metrics.update(
                            {
                                "loss": main_loss_step.item() / (t_total * recursions),
                                "reg": reg_loss_step.item() / (t_total * recursions),
                                "factor": factor_step.mean().item() / (t_total * recursions),
                            }
                        )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pbar.set_description(
                    f"Epoch {epoch+1}/{self.optim.epochs} - @ {metrics['factor'] / max(1, i+1):.4f} - Loss: {metrics['loss'] / max(1, i+1):.4f} - Reg: {metrics['reg'] / max(1, i+1):.4f}"
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
                    n_targets = self.data.info.n_targets
                    if isinstance(n_targets, tuple):
                        task_accs = [evaluation.report.get(f"task_{i}_acc", 0) for i in range(len(n_targets))]
                        val_score = np.mean(task_accs)
                        print(
                            f"* Val Loss: {val_metrics['loss']:.4f} - Task Accs: {[f'{a:.4f}' for a in task_accs]}"
                        )
                    else:
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

                scheduler.step(val_score)

                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_model = deepcopy(model)
                    print("-- New best model --")

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
                        n_targets = self.data.info.n_targets
                        if isinstance(n_targets, tuple):
                            splits = list(n_targets)
                            output_parts = torch.split(output, splits, dim=-1)
                            proba_parts = []
                            pred_parts = []
                            for o in output_parts:
                                p = torch.softmax(o, dim=-1)
                                proba_parts.append(p)
                                pred_parts.append(torch.argmax(p, dim=-1))
                            proba = torch.cat(proba_parts, dim=-1).cpu()
                            predicted = torch.stack(pred_parts, dim=-1).cpu()
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
            n_targets = self.data.info.n_targets
            if isinstance(n_targets, tuple):
                report = {}
                splits = list(n_targets)
                cum_splits = np.cumsum(splits)
                y_true_task = np.split(y_true_np, cum_splits[:-1], axis=1)
                y_proba_task = np.split(y_proba_np, cum_splits[:-1], axis=1)
                for i, (yt, ypr, dim) in enumerate(
                    zip(y_true_task, y_proba_task, splits)
                ):
                    yt_class = np.argmax(yt, axis=1)
                    yp_class = y_pred_np[:, i]
                    acc = accuracy_score(yt_class, yp_class)
                    if dim == 2:
                        try:
                            auc_val = roc_auc_score(yt_class, ypr[:, 1], average="macro")
                            report[f"task_{i}_auc"] = auc_val
                        except ValueError:
                            pass
                    report[f"task_{i}_acc"] = acc
            else:
                if n_targets == 2:
                    auc = roc_auc_score(
                        y_true_np, [p for _, p in y_proba_np], average="macro"
                    )
                else:
                    auc = roc_auc_score(
                        y_true_np, y_proba_np, multi_class="ovr", average="macro"
                    )

                kwargs = {"average": None, "zero_division": 0}
                acc = accuracy_score(y_true_np, y_pred_np)
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
