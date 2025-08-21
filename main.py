import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from functools import partial
import argparse
import sys
import os

from datasets import (
    load_time_series_dataset,
    load_air_quality_dataset,
    load_mnist_dataset,
    load_fairface_dataset,
    load_credit_risk_dataset,
    load_credit_card_fraud_dataset,
    load_uci_adult_dataset,
    load_playground_series_s5e3_dataset,
    load_obesity_classification_dataset,
    load_heart_failure_prediction_dataset,
    load_insurance_dataset,
    load_boston_house_price_dataset,
    load_concrete_compressive_strength_dataset,
    load_energy_efficiency_dataset,
    load_custom_dataset,
)
from modules import Network, UnitBlock, RecurrentBlock, PaddedConv2d, HRMBlock
from regularization import ActivationRegularizer
from training import Evaluator, Trainer, Training, Task
from visualization import Visualizer
from copy import deepcopy


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping of dataset names to loading functions
DATASET_LOADERS = {
    "time_series": load_time_series_dataset,
    "air_quality": load_air_quality_dataset,
    "mnist": load_mnist_dataset,
    "fairface": load_fairface_dataset,
    "credit_risk": load_credit_risk_dataset,
    "credit_card_fraud": load_credit_card_fraud_dataset,
    "uci_adult": load_uci_adult_dataset,
    "playground_s5e3": load_playground_series_s5e3_dataset,
    "obesity": load_obesity_classification_dataset,
    "heart_failure": load_heart_failure_prediction_dataset,
    "insurance": load_insurance_dataset,
    "boston_house_price": load_boston_house_price_dataset,
    "concrete_compressive_strength": load_concrete_compressive_strength_dataset,
    "energy_efficiency": load_energy_efficiency_dataset,
}


def select_best_model(trainer: Trainer) -> tuple[nn.Module, list[float]]:
    """
    Selects the best model from the training generator.

    Args:
        trainer (Trainer): The trainer object.

    Returns:
        tuple[nn.Module, list[float]]: The best model and the history of validation scores.
    """
    best_model = None
    running_score = -float("inf")
    history = []

    for epoch, val_score, model in trainer():
        history.append(val_score)
        if best_model is None:
            running_score = val_score

        ratio = (0.99, 0.01)
        if val_score >= running_score:
            best_model = deepcopy(model)
            print("-- New best model --")
            ratio = (0.8, 0.2)

        running_score = running_score * ratio[0] + val_score * ratio[1]

    return best_model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate EZTrainer models on various datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Name of the dataset to use or path to a custom dataset file. Available presets: {', '.join(DATASET_LOADERS.keys())}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,  # Will be adjusted based on cnn_task and time_series
        help="Dimension of the hidden layers.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=16,
        help="Number of hidden layers.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,  # Will be adjusted based on time_series
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--reg_factor",
        type=float,
        default=1732e-6,  # No particular reason but works well
        help="Regularization factor.",
    )
    parser.add_argument(
        "--regularization_period",
        type=int,
        default=4,
        help="The full cycle for regularization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoaders. If not provided, determined by dataset config.",
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        default=None,
        help="Length of the generated time series dataset.",
    )

    args = parser.parse_args()

    # --- Configuration ---
    # Load the selected dataset
    dataset_identifier = args.dataset
    if os.path.exists(dataset_identifier):
        print(f"Loading custom dataset from: {dataset_identifier}")
        train_dataset, val_dataset, test_dataset, config = load_custom_dataset(
            dataset_identifier
        )
    elif dataset_identifier in DATASET_LOADERS:
        print(f"Loading preset dataset: {dataset_identifier}")
        if dataset_identifier == "time_series" and args.dataset_length:
            train_dataset, val_dataset, test_dataset, config = DATASET_LOADERS[
                dataset_identifier
            ](length=args.dataset_length)
        else:
            train_dataset, val_dataset, test_dataset, config = DATASET_LOADERS[
                dataset_identifier
            ]()
    else:
        print(f"Error: Dataset '{dataset_identifier}' not found.")
        print(f"Available presets: {', '.join(DATASET_LOADERS.keys())}")
        sys.exit(1)

    batch_size = args.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    task = Task(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    print("Dataset loaded successfully.")

    # Model parameters
    inner_module = PaddedConv2d if config.cnn else nn.Linear
    model_args = {
        "input_dim": config.input_dim,
        "hidden_dim": (
            args.hidden_dim // (7 * config.cnn + 1) // (7 * config.timeseries + 1)
        ),
        "n_hidden": args.n_hidden,
        "output_dim": config.n_targets,
        "inner_module": inner_module,
        "inner_block": HRMBlock if config.timeseries else UnitBlock,
        "activation": torch.tanh if config.classify else torch.nn.SiLU(),
        "activation_params": {},
        "collapse_output": True,
        "dtype": torch.float32,
    }

    # Training parameters
    lr = args.lr / (9 * config.timeseries + 1)
    train_args = {
        "task": task,
        "criterion": (
            nn.CrossEntropyLoss(reduction="none") if config.classify else nn.MSELoss()
        ),
        "epochs": args.epochs,
        "regularizer": partial(ActivationRegularizer, module_type=inner_module),
        "reg_factor": args.reg_factor,
        "regularization_period": args.regularization_period,
    }

    # --- Model Training ---
    model = Network(**model_args)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_root = "logs"
    activation_name = "tanh" if config.classify else "silu"
    log_name = f"main_model_{activation_name}_{lr}"
    logger = TensorBoardLogger(log_root, log_name)

    print("Training model...")

    training = Training(
        model=model,
        optimizer=optimizer,
        **train_args,
        device=device,
    )

    trainer = Trainer(training=training, logger=logger)
    evaluator = Evaluator(training=training, logger=logger)

    best_model, history = select_best_model(trainer)

    print("Finished training model.")

    # --- Model Evaluation ---
    if best_model:
        print("\nEvaluating the best model on the test set...")
        evaluator.training.model = best_model
        evaluation = evaluator()

        if config.classify:
            acc, auc, f1 = evaluation.metrics
            class_report = evaluation.report
            print(
                f"Test Metrics - Accuracy: {acc:.4f}, AUC: {auc:.4f}, F1-score: {f1:.4f}"
            )
            print("Classification Report:")
            for metric_name, scores in class_report.items():
                print(f"  {metric_name}: {scores}")
        else:
            mae, mse = evaluation.metrics
            print(f"Test Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}")

    # --- Visualization ---
    if best_model:
        print("\nGenerating visualizations...")
        visualizer = Visualizer(evaluator=evaluator)
        visualizer.visualize()
        visualizer.plot_node_distribution()
        visualizer.plot_learning_curve(history)
        if config.classify:
            visualizer.visualize_tsne()  # t-SNE can be computationally expensive and might not be suitable for all datasets/environments
            if config.cnn:
                visualizer.plot_misclassified_examples(
                    num_examples=72
                )  # Example: plot 72 misclassified examples


if __name__ == "__main__":
    main()
