import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from functools import partial
import argparse
import sys

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
)
from modules import Network, UnitBlock, RecurrentBlock, PaddedConv2d
from regularization import ActivationRegularizer
from training import Evaluator, Trainer, Training, Task
from visualization import Visualizer

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


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate EZTrainer models on various datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_LOADERS.keys(),
        help=f"Name of the dataset to use. Available datasets: {', '.join(DATASET_LOADERS.keys())}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=(v_ := 50),
        help=f"Number of training epochs (default: {v_}).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=(v_ := 512),  # Will be adjusted based on cnn_task and time_series
        help=f"Dimension of the hidden layers (default: {v_}).",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=(v_ := 16),
        help=f"Number of hidden layers (default: {v_}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=(v_ := 1e-4),  # Will be adjusted based on time_series
        help=f"Learning rate for the optimizer (default: {v_}).",
    )
    parser.add_argument(
        "--reg_factor",
        type=float,
        default=(v_ := 17e-4),
        help=f"Regularization factor (default: {v_}).",
    )
    parser.add_argument(
        "--sparsity_period",
        type=int,
        default=(v_ := 5),
        help=f"Sparsity period for regularization (default: {v_}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=(v := 32),
        help=f"Batch size for DataLoaders. If not provided, determined by dataset config (default: {v_}).",
    )

    args = parser.parse_args()

    # --- Configuration ---
    # Load the selected dataset
    dataset_name = args.dataset
    if dataset_name not in DATASET_LOADERS:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: {', '.join(DATASET_LOADERS.keys())}")
        sys.exit(1)

    print(f"Loading dataset: {dataset_name}")
    # Handle potential arguments for specific dataset loaders if necessary
    # For now, assuming no extra args needed for the selected dataset loaders
    train_dataset, val_dataset, test_dataset, config = DATASET_LOADERS[dataset_name]()

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
        "inner_block": RecurrentBlock if config.timeseries else UnitBlock,
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
        "num_epochs": args.epochs,
        "regularizer": partial(ActivationRegularizer, module_type=inner_module),
        "reg_factor": args.reg_factor,
        "sparsity_period": args.sparsity_period,
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

    best_model = trainer()

    print("Finished training model.")

    # --- Model Evaluation ---
    if best_model:
        print("\nEvaluating the best model on the test set...")
        evaluation = evaluator()

        if config.classify:
            acc, auc = evaluation.metrics
            class_report = evaluation.report
            print(f"Test Metrics - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
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
        if config.classify:
            visualizer.visualize_tsne()  # t-SNE can be computationally expensive and might not be suitable for all datasets/environments
            if config.cnn:
                visualizer.plot_misclassified_examples(
                    num_examples=72
                )  # Example: plot 72 misclassified examples


if __name__ == "__main__":
    main()
