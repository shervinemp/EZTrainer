from dataclasses import dataclass
import os
import subprocess
import zipfile
from typing import List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset
from imblearn.over_sampling import RandomOverSampler
import torch
from torch.utils.data import DataLoader


@dataclass
class TaskConfig:
    name: str
    input_dim: int
    n_targets: Union[int, Tuple[int, ...]]
    classify: bool
    cnn: bool
    timeseries: bool


@dataclass
class Task:
    config: TaskConfig
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None


def download_file_from_drive(file_id, local_path, *args, **kwargs):
    """
    Downloads a single file from Google Drive using its file ID.

    Args:
        file_id (str): The ID of the file in Google Drive.
        local_path (str): The desired local path to save the file.
    """
    try:
        import gdown
    except ImportError:
        subprocess.run(["pip", "install", "gdown"], check=True)
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, local_path, *args, quiet=False, **kwargs)
        print(f"File successfully downloaded to '{local_path}'.")
    except Exception as e:
        print(f"Error downloading file: {e}")


def _download_and_extract_kaggle_dataset(
    download_url: str, filenames_to_extract: List[str], zip_file_name: str
):
    """
    Downloads a zip file from a given URL and extracts specified files.

    Args:
        download_url (str): The URL of the zip file to download.
        filenames_to_extract (List[str]): A list of filenames to extract from the zip.
        zip_file_name (str): The name to save the downloaded zip file as.
    """
    # Check if all required files already exist
    if all(os.path.exists(filename) for filename in filenames_to_extract):
        print(f"Dataset files {filenames_to_extract} already exist. Skipping download.")
        return

    print(f"Downloading dataset from {download_url}...")
    try:
        # Use curl to download the zip file
        subprocess.run(["curl", "-L", "-o", zip_file_name, download_url], check=True)
        print("Download complete. Extracting files...")
        # Use zipfile to extract the necessary files
        with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
            f_ = filenames_to_extract or zip_ref.namelist()
            for filename in f_:
                zip_ref.extract(filename)
        print("Extraction complete.")
        os.remove(zip_file_name)  # Clean up the zip file
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}")
        # Exit or raise an error if dataset cannot be loaded
        raise FileNotFoundError(
            f"Could not load dataset. Download or extraction failed: {e}"
        )


def dataset_from_numpy(
    X: np.ndarray,
    y: np.ndarray,
    classify: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.2,
    oversample: bool = False,
    imputation: bool = False,
    n_partitions: Optional[int] = None,
    partition_overlap: float = 0.0,
    random_state: int = 42,
) -> Tuple[Tuple[TensorDataset, TensorDataset], TensorDataset]:
    """
    Creates PyTorch datasets with optional supersampling.

    Args:
        X: NumPy array of features.
        y: NumPy array of labels.
        classify: Boolean indicating if it's a classification task.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the training dataset to include in the validation split.
        oversample: Boolean indicating whether to apply random oversampling (for classification).
        imputation: Boolean indicating whether to perform mean imputation for NaN values.
        n_partitions: Optional number of partitions for time series data.
        partition_overlap: Overlap between partitions (if n_partitions is not None).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_dataset, test_dataset), val_dataset.
    """
    assert X.shape[0] == y.shape[0], "X and y must have the same number of rows."
    assert 0 <= test_size <= 1, "test_size must be between 0 and 1."
    assert 0 <= val_size <= 1, "val_size must be between 0 and 1."
    assert 0 <= partition_overlap <= 1, "partition_overlap must be between 0 and 1."

    if imputation:
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # Normalize data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if not classify:
        scaler = StandardScaler()
        y = scaler.fit_transform(y.reshape(-1, 1))

    if n_partitions is not None:
        n_samples = int(X.shape[0] * (1 + partition_overlap) // n_partitions)
        step = int(X.shape[0] // n_partitions)

        # Create views using sliding_window_view with conditional stride
        X = np.lib.stride_tricks.sliding_window_view(X, (n_samples, *X.shape[1:]))[
            ::step
        ]
        y = np.lib.stride_tricks.sliding_window_view(y, (n_samples, *y.shape[1:]))[
            ::step
        ]

        X = X.squeeze(1)[:n_partitions]
        y = y.squeeze(1)[:n_partitions]

    # Handle potential empty splits when stratifying with very small datasets
    stratify_train = y if classify and n_partitions is None else None
    if stratify_train is not None and len(np.unique(stratify_train)) < 2:
        stratify_train = None  # Cannot stratify with only one class

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_train
    )

    stratify_val = y_train if classify and n_partitions is None else None
    if stratify_val is not None and len(np.unique(stratify_val)) < 2:
        stratify_val = None  # Cannot stratify with only one class

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_val,
    )

    # Supersampling (if classification)
    if classify and oversample:
        ros = RandomOverSampler(random_state=random_state)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    # Convert to PyTorch tensors
    target_dtype: torch.dtype = torch.int64 if classify else torch.float32

    X_train_tensor: torch.Tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor: torch.Tensor = torch.tensor(y_train, dtype=target_dtype)
    X_test_tensor: torch.Tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor: torch.Tensor = torch.tensor(y_test, dtype=target_dtype)
    X_val_tensor: torch.Tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor: torch.Tensor = torch.tensor(y_val, dtype=target_dtype)

    # Create TensorDatasets
    train_dataset: TensorDataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset: TensorDataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset: TensorDataset = TensorDataset(X_val_tensor, y_val_tensor)

    return (train_dataset, test_dataset), val_dataset
