import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
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
from training import DataParams, Evaluator, OptimParams, Trainer, TrainParams
from visualization import Visualizer

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
        description="Train and evaluate EZTrainer models on various datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=1e-4,  # Will be adjusted if time_series
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--reg_factor",
        type=float,
        default=1e-2,
        help="Regularization factor.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoaders. If not provided, determined by dataset config.",
    )
    parser.add_argument(
        "--min_batches",
        type=int,
        default=10,
        help="Minimum number of batches per epoch, achieved by repeating the dataloader.",
    )
    parser.add_argument(
        "--n_recurse",
        type=int,
        default=1,
        help="Number of recursions for each data point. Forces time-series behaviour.",
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name not in DATASET_LOADERS:
        print(f"Error: Dataset '{dataset_name}' not found.")
        print(f"Available datasets: {', '.join(DATASET_LOADERS.keys())}")
        sys.exit(1)

    print(f"Loading dataset: {dataset_name}")

    data_info = DATASET_LOADERS[dataset_name]()
    batch_size = args.batch_size
    n_recurse = args.n_recurse

    train_loader = DataLoader(
        data_info.trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        data_info.valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        data_info.testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    data_params = DataParams(
        info=data_info,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        repeats=(args.min_batches - 1) // len(train_loader) + 1,
        recursions=n_recurse,
    )
    print("Dataset loaded successfully.")
    print(
        f"{len(data_info.trainset)=}, {len(data_info.testset)=}, {len(data_info.valset)=}"
    )

    hidden_dim = (
        args.hidden_dim
        // (7 * data_info.is_image + 1)
        // (7 * data_info.is_timeseries + 1)
    )
    inner_module = PaddedConv2d if data_info.is_image else nn.Linear
    inner_block = (
        RecurrentBlock if (data_info.is_timeseries or n_recurse > 1) else UnitBlock
    )
    activation = torch.tanh if data_info.is_classify else torch.nn.SiLU()
    model_args = {
        "input_dim": data_info.input_dim,
        "hidden_dim": hidden_dim,
        "n_hidden": args.n_hidden,
        "output_dim": data_info.n_targets,
        "inner_module": inner_module,
        "inner_block": inner_block,
        "activation": activation,
        "activation_params": {},
        "collapse_output": True,
        "dtype": torch.float32,
    }
    model = Network(**model_args)

    lr_scale = 10 if data_info.is_timeseries else 1
    lr = args.lr * lr_scale

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr)

    log_root = "logs"
    activation_name = "tanh" if data_info.is_classify else "silu"
    log_name = f"main_model_{activation_name}_{lr}"
    logger = TensorBoardLogger(log_root, log_name)

    criterion = (
        nn.CrossEntropyLoss(reduction="none") if data_info.is_classify else nn.MSELoss()
    )

    optim_params = OptimParams(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        reg_factor=args.reg_factor,
        device=device,
    )

    print("Training model...")

    params = TrainParams(optim=optim_params, data=data_params)
    trainer = Trainer(params=params, logger=logger)
    evaluator = Evaluator(params=params, logger=logger)

    best_model = trainer()

    print("Finished training model.")

    # --- Model Evaluation ---
    if best_model:
        print("\nEvaluating the best model on the test set...")
        evaluation = evaluator("test")
        print("Report:")
        for metric_name, scores in evaluation.report.items():
            print(f"  {metric_name}: {scores}")

    # --- Visualization ---
    if best_model:
        print("\nGenerating visualizations...")
        visualizer = Visualizer(evaluator=evaluator)
        visualizer.visualize()
        visualizer.plot_node_distribution()
        if data_info.is_classify:
            visualizer.visualize_tsne()  # t-SNE can be computationally expensive and might not be suitable for all datasets/environments
            if data_info.is_image:
                visualizer.plot_misclassified_examples(
                    num_examples=72
                )  # Example: plot 72 misclassified examples


if __name__ == "__main__":
    main()
