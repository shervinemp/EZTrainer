import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.manifold import TSNE
from collections import defaultdict
from data_utils import Task
from modules import Affine
from typing import List, Tuple, Dict

from training import Evaluator


class Visualizer:
    """
    Visualizes model performance and data characteristics.
    """

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def visualize(self):
        """
        Visualizes the model's performance and data characteristics.
        """
        task = self.evaluator._tr.task
        config = task.config
        evaluation = self.evaluator()

        if config.classify:
            cm = confusion_matrix(
                evaluation.y_true, evaluation.y_pred, normalize="true"
            )
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(
                config.n_targets
                if isinstance(config.n_targets, int)
                else sum(config.n_targets)
            )
            plt.xticks(tick_marks, tick_marks)
            plt.yticks(tick_marks, tick_marks)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.show()

            if isinstance(config.n_targets, int) and config.n_targets == 2:
                if len(np.unique(evaluation.y_true)) > 1:
                    fpr, tpr, _ = roc_curve(
                        evaluation.y_true, [p[1] for p in evaluation.y_proba]
                    )
                    roc_auc = auc(fpr, tpr)
                    plt.figure()
                    plt.plot(
                        fpr,
                        tpr,
                        color="darkorange",
                        lw=2,
                        label=f"ROC curve (area = {roc_auc:.2f})",
                    )
                    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend(loc="lower right")
                    plt.show()
                else:
                    print(
                        "Cannot plot ROC curve with only one class present in true labels."
                    )

        else:
            plt.figure(figsize=(8, 6))
            plt.scatter(evaluation.y_true, evaluation.y_pred, alpha=0.5)
            plt.plot(
                [min(evaluation.y_true), max(evaluation.y_true)],
                [min(evaluation.y_true), max(evaluation.y_true)],
                color="red",
                linestyle="--",
            )
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predicted vs. Actual Values")
            plt.grid(True)
            plt.show()

    def plot_node_distribution(self):
        """
        Plots the distribution of the means of the absolute output of each node in the linear layers
        and the distribution of weights in affine layers.
        """

        def get_activation(name: str):
            def hook(
                model: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
            ):
                if isinstance(model, nn.Linear):
                    linear_outputs[name].append(output.abs().mean(dim=0).cpu().numpy())

            return hook

        task = self.evaluator._tr.task
        config = task.config
        model = self.evaluator._tr.model

        hooks: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))

        linear_outputs: Dict[str, List[np.ndarray]] = defaultdict(list)

        model.eval()
        with torch.no_grad():
            try:
                inputs, _ = next(iter(task.test_loader))

                if not config.timeseries:
                    inputs = inputs.unsqueeze(1)

                inputs = inputs.to(self.evaluator._tr.device)
                _ = model(inputs[:, 0, ...])
            except StopIteration:
                print(
                    "Warning: Could not get a batch from data_loader for node distribution plotting."
                )
                return

        for hook in hooks:
            hook.remove()

        # Collect and plot linear means if available
        if linear_outputs:
            linear_means = np.array(
                [x for k, v in linear_outputs.items() for x in np.mean(v, axis=0)]
            )
            if linear_means.size > 0:
                self._plot_linear_weights(linear_means)
            else:
                print("No linear layer outputs collected for plotting.")
        else:
            print("No linear layers found in the model for node distribution plotting.")

        affine_weights = []
        for _, m in model.named_modules():
            if isinstance(m, Affine):
                affine_weights.extend(m.weight.detach().abs().cpu().flatten().tolist())
        affine_weights = np.array(affine_weights)

        if affine_weights.size > 0:
            self._plot_affine_means(affine_weights)
        else:
            print("No affine layers found in the model for node distribution plotting.")

        # Plot weights distribution (only for Linear layers)
        self._plot_weights_distribution()

    def _plot_weights_distribution(self):
        """Plots the distribution of weights in linear layers."""
        model = self.evaluator._tr.model
        plt.figure(figsize=(10, 6))
        weights = np.concatenate(
            [
                w.flatten()
                for w in (
                    m.weight.detach().abs().cpu()
                    for _, m in model.named_modules()
                    if isinstance(m, nn.Linear)
                )
            ]
        )
        if weights.size > 0:
            bins = np.linspace(min(weights), max(weights), num=120, endpoint=True)
            hist, _ = np.histogram(weights, bins=bins)
            plt.bar(bins[:-1], hist, width=np.diff(bins), align="edge")
            plt.xlabel("Weight Value (log2 scale)")
            plt.ylabel("Number of Weights (log10 scale)")
            plt.title("Distribution of Weights")
            plt.yscale("log")
            plt.grid(True)
            if len(weights) > 0:
                plt.xticks(
                    bins[:: (s_ := max(1, len(bins) // 10))],
                    [
                        f"{x:.2e}\n{c:.2%}"
                        for x, c in zip(
                            bins[::s_], (np.cumsum(hist) / len(weights))[::s_]
                        )
                    ],
                )
            plt.tight_layout()
            plt.show()
        else:
            print("No weights found in linear layers for plotting.")

    def _plot_linear_weights(self, linear_weights):
        """Plots the distribution of node output means in linear layers."""
        plt.figure(figsize=(10, 6))
        if linear_weights.size > 0:
            if np.min(linear_weights) == np.max(linear_weights):
                bins = np.array([np.min(linear_weights), np.min(linear_weights) + 1e-6])
            else:
                bins = np.logspace(
                    np.log10(max(1e-9, np.min(linear_weights))),
                    np.log10(np.max(linear_weights)),
                    num=120,
                    endpoint=True,
                    base=10,
                )
            hist, _ = np.histogram(linear_weights, bins=bins)
            plt.bar(bins[:-1], hist, width=np.diff(bins), align="edge")
            plt.xlabel("Mean of Absolute Linear Output (log10 scale)")
            plt.ylabel("Number of Linear Layers (log10 scale)")
            plt.title("Distribution of Node Output Means")
            plt.xscale("log")
            plt.yscale("log")
            plt.grid(True)
            if len(linear_weights) > 0:
                plt.xticks(
                    bins[:: (s_ := max(1, len(bins) // 10))],
                    [
                        f"{x:.2e}\n{c:.2%}"
                        for x, c in zip(
                            bins[::s_], (np.cumsum(hist) / len(linear_weights))[::s_]
                        )
                    ],
                )
            plt.tight_layout()
            plt.show()
        else:
            print("No linear layer outputs collected for plotting.")

    def _plot_affine_means(self, affine_weights):
        """Plots the distribution of affine layer weights."""
        plt.figure(figsize=(10, 6))
        if affine_weights.size > 0:
            if np.min(affine_weights) == np.max(affine_weights):
                bins = np.array([np.min(affine_weights), np.min(affine_weights) + 1e-6])
            else:
                bins = np.logspace(
                    np.log2(max(1e-9, np.min(affine_weights))),
                    np.log2(np.max(affine_weights)),
                    num=120,
                    endpoint=True,
                    base=2,
                )
            hist, _ = np.histogram(affine_weights, bins=bins)
            plt.bar(bins[:-1], hist, width=np.diff(bins), align="edge")
            plt.xlabel("Mean of Absolute Affine Weights (log2 scale)")
            plt.ylabel("Number of Affine Layers (log10 scale)")
            plt.title("Distribution of Affine Layer Means")
            plt.xscale("log", base=2)
            plt.yscale("log")
            plt.grid(True)
            if len(affine_weights) > 0:
                plt.xticks(
                    bins[:: (s_ := max(1, len(bins) // 10))],
                    [
                        f"{x:.2e}\n{c:.2%}"
                        for x, c in zip(
                            bins[::s_], (np.cumsum(hist) / len(affine_weights))[::s_]
                        )
                    ],
                )
            plt.tight_layout()
            plt.show()
        else:
            print("No affine layer weights collected for plotting.")

    def plot_misclassified_examples(
        self,
        num_examples: int = 10,
    ):
        """
        Plots a specified number of misclassified examples based on provided true labels, predictions, and probabilities.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels or values.
            y_proba (np.ndarray): Predicted probabilities (for classification).
            num_examples (int, optional): The number of misclassified examples to plot.
                Defaults to 10.
        """
        evaluation = self.evaluator()

        y_true = evaluation.y_true
        y_pred = evaluation.y_pred
        y_proba = evaluation.y_proba

        if not self.classify:
            print("Misclassified examples plotting is only for classification tasks.")
            return

        misclassified_indices = np.where(y_true != y_pred)[0]
        if len(misclassified_indices) == 0:
            print("No misclassified examples found.")
            return

        # Get the original images corresponding to the misclassified indices

        misclassified_images: List[np.ndarray] = []
        current_index = 0
        for inputs, _ in self.data_loader:
            inputs = inputs.cpu().numpy()
            batch_size = inputs.shape[0]
            indices_in_batch = (
                misclassified_indices[
                    (misclassified_indices >= current_index)
                    & (misclassified_indices < current_index + batch_size)
                ]
                - current_index
            )

            for idx in indices_in_batch:
                misclassified_images.append(inputs[idx])
                if len(misclassified_images) == num_examples:
                    break

            current_index += batch_size
            if len(misclassified_images) == num_examples:
                break

        if not misclassified_images:
            print("No misclassified examples found.")
            return

        misclassified_labels = y_true[misclassified_indices][:num_examples]
        misclassified_predictions = y_pred[misclassified_indices][:num_examples]
        probas = y_proba[misclassified_indices][:num_examples]

        num_rows = int(np.ceil(num_examples / 6))
        num_cols = 6

        plt.figure(figsize=(10 * num_cols / 5, 5 * num_rows / 2))
        for i, img in enumerate(misclassified_images[:num_examples]):
            # Check if the data point is suitable for image plotting
            if img.ndim in [2, 3]:
                plt.subplot(num_rows, num_cols, i + 1)
                if img.ndim == 2:  # Grayscale image
                    plt.imshow(img, cmap="gray")
                elif img.shape[0] == 1:  # Grayscale image with channel dimension
                    plt.imshow(img.squeeze(0), cmap="gray")
                else:  # Color image
                    plt.imshow(np.transpose(img, (1, 2, 0)))

                plt.title(
                    f"T: {(l_:=misclassified_labels[i])} ({probas[i][l_]:.2f}), P: {(o:=misclassified_predictions[i])} ({probas[i][o]:.2f})"
                )
                plt.axis("off")
            else:
                # Print feature values, true label, and predicted label for tabular data
                print(f"Misclassified example {i}:")
                print(f"  Features: {img}")
                print(f"  True Label: {misclassified_labels[i]}")
                print(f"  Predicted Label: {misclassified_predictions[i]}")

        plt.tight_layout()
        plt.show()

    def visualize_tsne(self):
        """
        Visualizes the data using t-SNE with true and predicted labels.
        """
        task = self.evaluator._tr.task
        config = task.config
        evaluation = self.evaluator()

        y_true = evaluation.y_true
        y_pred = evaluation.y_pred
        y_proba = evaluation.y_proba

        # Get embeddings from the data loader
        all_embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for input, _ in task.test_loader:
                input = input.cpu().numpy().reshape(len(input), -1)
                all_embeddings.extend(input)

        all_embeddings = np.array(all_embeddings)

        if all_embeddings.shape[0] < 2:
            print("Not enough samples to perform t-SNE.")
            return

        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(all_embeddings)

        tsne_all = embeddings_tsne[: len(all_embeddings)]

        plt.figure(figsize=(10, 8))
        for label in np.unique(y_true):
            indices = np.where(y_true == label)[0]
            plt.scatter(
                tsne_all[indices, 0], tsne_all[indices, 1], label=f"True: {label}"
            )

        plt.legend()
        plt.title("t-SNE Visualization of Test Data with True Labels")
        plt.show()

        plt.figure(figsize=(10, 8))
        for label in np.unique(y_pred):
            indices = np.where(y_pred == label)[0]
            plt.scatter(
                tsne_all[indices, 0], tsne_all[indices, 1], label=f"Predicted: {label}"
            )

        plt.legend()
        plt.title("t-SNE Visualization of Test Data with Predicted Labels")
        plt.show()

        if config.classify:
            plt.figure(figsize=(10, 8))
            for label in np.unique(y_true):
                indices = np.where(y_true == label)[0]
                if y_proba.shape[1] > label:
                    plt.scatter(
                        tsne_all[indices, 0],
                        tsne_all[indices, 1],
                        c=y_proba[indices, label],
                        label=f"True: {label}",
                    )
            plt.colorbar(label="Probability")
            plt.legend()
            plt.title(
                "t-SNE Visualization of Test Data with True Labels (Colored by Probability)"
            )
            plt.show()

    def display_metrics(self):
        """
        Evaluates the model using the provided Evaluator and displays the metrics.
        """
        evaluation = self.evaluator()

        metrics = evaluation["metrics"]
        report = evaluation["report"]

        print("\n--- Evaluation Metrics ---")
        if self.classify:
            print(f"Accuracy: {metrics[0]:.4f}")
            print(f"AUC: {metrics[1]:.4f}")
            print("Classification Report:")
            for metric_name, values in report.items():
                print(f"  {metric_name.capitalize()}: {values}")
        else:
            print(f"MAE: {metrics[0]:.4f}")
            print(f"MSE: {metrics[1]:.4f}")
        print("--------------------------\n")
