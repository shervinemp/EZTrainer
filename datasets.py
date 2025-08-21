import os
import torch
import numpy as np
import pandas as pd
import io
import zipfile
import requests
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ucimlrepo import fetch_ucirepo

from data_utils import (
    dataset_from_numpy,
    download_file_from_drive,
    _download_and_extract_kaggle_dataset,
    TaskConfig,
)


class FairFaceDataset(Dataset):
    """Custom Dataset for the FairFace dataset."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str = "",
        transform: transforms.Compose = None,
        one_hot: bool = True,
        cache: bool = False,
    ):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            one_hot (bool): Whether to one-hot encode the labels.
            cache (bool): Whether to cache images in memory.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_cache: dict[str, Image.Image] | None = {} if cache else None
        self._process_labels(one_hot)

    def _process_labels(self, one_hot: bool) -> None:
        """Processes the categorical labels, applying one-hot encoding if specified."""
        self.categorical_cols = self.data_frame.select_dtypes(
            exclude=[np.number]
        ).columns.drop("file")
        self.targets = self.data_frame[self.categorical_cols]
        self.encoders: list[LabelEncoder] = [
            LabelEncoder().fit(self.targets[col]) for col in self.targets.columns
        ]
        self.targets = pd.DataFrame(
            {
                col: encoder.transform(self.targets[col])
                for col, encoder in zip(self.targets.columns, self.encoders)
            }
        )
        if one_hot:
            self.targets = pd.get_dummies(self.targets, columns=self.targets.columns)
            self.one_hot_columns: list[str] = (
                self.targets.columns.to_list()
            )  # store columns here
        else:
            self.one_hot_columns: list[str] = self.targets.columns.to_list()
        self.targets = torch.tensor(self.targets.values, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[Image.Image, torch.Tensor]:
        """
        Retrieves an image and its corresponding label from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Image.Image, torch.Tensor]: A tuple containing the image and its label.
        """
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = self._load_image(img_name)
        label = self.targets[idx]
        return image, label

    def _load_image(self, img_name: str) -> Image.Image:
        """
        Loads an image from the specified path, with optional caching and transformation.

        Args:
            img_name (str): The path to the image file.

        Returns:
            Image.Image: The loaded and potentially transformed image.
        """
        im_cache = self.image_cache
        if im_cache and img_name in im_cache:
            return im_cache[img_name]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if im_cache:
            im_cache[img_name] = image
        return image

    def decode_label(self, encoded_label_tensor: torch.Tensor) -> dict[str, str | None]:
        """
        Decodes an encoded label tensor back to its original categorical values.

        Args:
            encoded_label_tensor (torch.Tensor): The encoded label tensor.

        Returns:
            dict[str, str | None]: A dictionary mapping categorical column names to their decoded values.
        """
        encoded_label = encoded_label_tensor.numpy()
        if len(self.categorical_cols) > 1:
            temp_df = pd.DataFrame(
                encoded_label.reshape(1, -1), columns=self.one_hot_columns
            ).astype(int)
            temp_df = temp_df.loc[:, (temp_df != 0).any(axis=0)]
            decoded_data = {
                col: self._decode_column(temp_df, col) for col in self.categorical_cols
            }
        else:
            decoded_data = {
                self.categorical_cols[0]: self.encoders[0].inverse_transform(
                    [np.argmax(encoded_label)]
                )[0]
            }
        return decoded_data

    def _decode_column(self, temp_df: pd.DataFrame, col: str) -> str | None:
        """
        Decodes a single categorical column from a temporary DataFrame.

        Args:
            temp_df (pd.DataFrame): The temporary DataFrame containing the encoded column.
            col (str): The name of the categorical column to decode.

        Returns:
            str | None: The decoded categorical value, or None if the column is not found.
        """
        col_names = [c for c in temp_df.columns if col in c]
        if col_names:
            decoded_val = temp_df[col_names].idxmax(axis=1).values[0].split("_")[-1]
            return self.encoders[self.categorical_cols.get_loc(col)].inverse_transform(
                [int(decoded_val)]
            )[0]
        return None


# Dataset loading functions (adapted from the notebook)
def load_time_series_dataset(
    length: int = 10000, n_features: int = 2, time_only: bool = False
) -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Generates a time series dataset for regression.
    If time_only is True, generates a dataset with only temporal dependency.

    Args:
        length (int): The total length of the time series.
        n_features (int): The number of features in the time series (if time_only is False).
        time_only (bool): If True, generates a dataset with only temporal dependency.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    if time_only:
        y = np.zeros(length)
        y[0] = np.random.rand()
        for t in range(1, length):
            y[t] = np.sin(2 * np.pi * y[t - 1] / 10 + t / 50) + np.cos(
                2 * np.pi * t / 500
            )
            y[t] += np.random.normal(0, 0.05)
        X = np.zeros((length - 1, 1))
        X[:, 0] = y[:-1]
        y = y[1:]
    else:
        X = np.zeros((length, n_features))
        y = np.zeros(length)
        X[0] = np.random.rand(n_features)
        for t in range(1, length):

            def inner_function(x: float, t: int) -> float:
                return np.sin(2 * np.pi * x / 10 + t / 50) + np.cos(2 * np.pi * t / 500)

            for feature_index in range(n_features):
                X[t, feature_index] = inner_function(X[t - 1, feature_index], t)
                X[t, feature_index] += np.random.normal(0, 0.05)
            y[t] = inner_function(X[t, 0], t)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(
        X, y, classify=False, n_partitions=1000, partition_overlap=0.1
    )

    config = TaskConfig(
        name="Time Series",
        input_dim=X.shape[1],
        n_targets=1,
        classify=False,
        cnn=False,
        timeseries=True,
    )

    return train_dataset, test_dataset, val_dataset, config


def load_air_quality_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the Air Quality dataset for regression.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_file = zf.read("AirQualityUCI.csv")

    df = pd.read_csv(io.BytesIO(csv_file), delimiter=";").dropna(how="all")
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S", errors="coerce"
    )
    df["TimeOfDaySin"] = np.sin(2 * np.pi * df["DateTime"].dt.hour / 24)
    df["TimeOfDayCos"] = np.cos(2 * np.pi * df["DateTime"].dt.hour / 24)
    df["DayOfYearSin"] = np.sin(2 * np.pi * df["DateTime"].dt.dayofyear / 365)
    df["DayOfYearCos"] = np.cos(2 * np.pi * df["DateTime"].dt.dayofyear / 365)
    df = df.set_index("DateTime")
    df = df.drop(["Date", "Time", "Unnamed: 15", "Unnamed: 16"], axis=1)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.replace(",", ".").astype(float, errors="ignore")

    X = df.drop("CO(GT)", axis=1).values.astype(float)
    y = df["CO(GT)"].values.astype(float)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(
        X, y, classify=False, imputation=True, n_partitions=1000, partition_overlap=0.5
    )

    config = TaskConfig(
        name="Air Quality",
        input_dim=X.shape[1],
        n_targets=1,
        classify=False,
        cnn=False,
        timeseries=True,
    )

    return train_dataset, test_dataset, val_dataset, config


def load_mnist_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the MNIST dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    trainset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_split, val_split = torch.utils.data.random_split(
        trainset,
        [(split := int(0.8 * len(trainset))), len(trainset) - split],
    )
    train_dataset = train_split
    val_dataset = val_split
    test_dataset = testset

    config = TaskConfig(
        name="MNIST",
        input_dim=1,
        n_targets=10,
        classify=True,
        cnn=True,
        timeseries=False,
    )

    return train_dataset, test_dataset, val_dataset, config


def load_fashion_mnist_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the Fashion MNIST dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    trainset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_split, val_split = torch.utils.data.random_split(
        trainset,
        [(split := int(0.8 * len(trainset))), len(trainset) - split],
    )
    train_dataset = train_split
    val_dataset = val_split
    test_dataset = testset

    config = TaskConfig(
        name="Fashion MNIST",
        input_dim=1,
        n_targets=10,
        classify=True,
        cnn=True,
        timeseries=False,
    )

    return train_dataset, test_dataset, val_dataset, config


def load_cifar10_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the CIFAR-10 dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform
    )

    train_split, val_split = torch.utils.data.random_split(
        trainset,
        [(split := int(0.8 * len(trainset))), len(trainset) - split],
    )
    train_dataset = train_split
    val_dataset = val_split
    test_dataset = testset

    config = TaskConfig(
        name="CIFAR-10",
        input_dim=3,
        n_targets=10,
        classify=True,
        cnn=True,
        timeseries=False,
    )

    return train_dataset, test_dataset, val_dataset, config


def load_fairface_dataset(
    root_dir: str = ".",
) -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the FairFace dataset for multi-label classification.

    Args:
        root_dir (str): The root directory where the dataset files are located.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    file_ids = [
        "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86",
        "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH",
        "1wOdja-ezstMEp81tX1a-EYkFebev4h7D",
    ]
    local_file_path = (
        "/content/"  # Assuming running in Colab or similar env with /content/
    )

    for file_id in file_ids:
        download_file_from_drive(file_id, local_file_path)

    # Assuming the zip file is extracted to the current directory or root_dir
    # You might need to add unzip commands here if not handled externally
    # Example: subprocess.run(['unzip', '-qq', 'fairface-img-margin025-trainval.zip'], check=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = FairFaceDataset(
        csv_file="fairface_label_train.csv", root_dir=root_dir, transform=transform
    )
    val_dataset = FairFaceDataset(
        csv_file="fairface_label_val.csv", root_dir=root_dir, transform=transform
    )
    # Note: FairFace doesn't have a standard test set split in the provided files,
    # so we'll use the validation set as the test set for evaluation purposes in this example.
    test_dataset = val_dataset

    config = TaskConfig(
        name="FairFace",
        input_dim=3,
        n_targets=tuple(len(encoder.classes_) for encoder in train_dataset.encoders),
        classify=True,
        cnn=True,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_custom_dataset(
    file_path: str,
) -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads a custom dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    data = pd.read_csv(file_path)
    data = data.dropna()

    if "target" in data.columns:
        target_col = "target"
    elif "label" in data.columns:
        target_col = "label"
    else:
        target_col = data.columns[-1]

    X = data.drop(target_col, axis=1).values
    y = data[target_col].values

    # Infer task type
    if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 30:
        classify = False
        n_targets = 1 if len(y.shape) == 1 else y.shape[1]
    else:
        classify = True
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_targets = len(le.classes_)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(
        X, y, classify=classify
    )

    config = TaskConfig(
        name=f"Custom Dataset: {os.path.basename(file_path)}",
        input_dim=X.shape[1],
        n_targets=n_targets,
        classify=classify,
        cnn=False,  # Assuming no CNN for custom tabular data
        timeseries=False,  # Assuming no time series for custom tabular data
    )

    return train_dataset, val_dataset, test_dataset, config


def load_credit_risk_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the Credit Risk dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    # Download and extract the dataset if not already present
    filenames = ["payment_data.csv", "customer_data.csv"]
    zip_name = "credit-risk-classification-dataset.zip"
    download_url = "https://www.kaggle.com/api/v1/datasets/download/praveengovi/credit-risk-classification-dataset"
    _download_and_extract_kaggle_dataset(download_url, filenames, zip_name)

    data_payment = pd.read_csv(filenames[0], index_col="id")
    data_customer = pd.read_csv(filenames[1], index_col="id")

    data = pd.merge(data_payment, data_customer, left_index=True, right_index=True)
    data = data.dropna()

    data = pd.get_dummies(
        data,
        columns=["prod_code", "fea_1", "fea_3", "fea_5", "fea_6", "fea_7", "fea_9"],
    )
    report_date = pd.to_datetime(data["report_date"], format="%d/%m/%Y")
    update_date = pd.to_datetime(data["update_date"], format="%d/%m/%Y")

    data["report_date_sin"] = np.sin(2 * np.pi * report_date.dt.dayofyear / 365)
    data["report_date_cos"] = np.cos(2 * np.pi * report_date.dt.dayofyear / 365)
    data["report_date_year"] = report_date.dt.year
    data = data.drop("report_date", axis=1)

    data["update_date_sin"] = np.sin(2 * np.pi * update_date.dt.dayofyear / 365)
    data["update_date_cos"] = np.cos(2 * np.pi * update_date.dt.dayofyear / 365)
    data["update_date_year"] = update_date.dt.year
    data = data.drop("update_date", axis=1)

    X = data.drop("label", axis=1).values.astype(float)
    y = data["label"].values.astype(int)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(X, y)

    config = TaskConfig(
        name="Credit Risk",
        input_dim=X.shape[1],
        n_targets=2,
        classify=True,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_credit_card_fraud_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the Credit Card Fraud Detection dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    # Download and extract the dataset if not already present
    filenames = ["creditcard.csv"]
    zip_name = "creditcardfraud.zip"
    download_url = (
        "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
    )
    _download_and_extract_kaggle_dataset(download_url, filenames, zip_name)

    df = pd.read_csv(filenames[0])
    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(X, y)

    config = TaskConfig(
        name="Credit Card Fraud",
        input_dim=X.shape[1],
        n_targets=2,
        classify=True,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_uci_adult_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the UCI Adult dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    data = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)
    data = data.dropna()

    data["income"] = LabelEncoder().fit_transform(data["income"])

    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    data = pd.get_dummies(data, columns=categorical_features)

    X = data.drop("income", axis=1).values.astype(float)
    y = data["income"].values.astype(int)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(X, y)

    config = TaskConfig(
        name="UCI Adult",
        input_dim=X.shape[1],
        n_targets=2,
        classify=True,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_playground_series_s5e3_dataset() -> (
    tuple[Dataset, Dataset, Dataset, TaskConfig]
):
    """
    Loads the Playground Series S5E3 dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    # Download and extract the dataset if not already present
    filenames = ["train.csv"]
    zip_name = "playground-series-s5e3.zip"
    download_url = (
        "https://www.kaggle.com/api/v1/datasets/download/kaggle/playground-series-s5e3"
    )
    _download_and_extract_kaggle_dataset(download_url, filenames, zip_name)

    data = pd.read_csv(filenames[0], index_col="id")
    data = data.dropna()

    data["day_sin"] = np.sin(2 * np.pi * data["day"] / 365)
    data["day_cos"] = np.cos(2 * np.pi * data["day"] / 365)
    data = data.drop("day", axis=1)

    X = data.drop("rainfall", axis=1).values.astype(float)
    y = data["rainfall"].values.astype(int)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(X, y)

    config = TaskConfig(
        name="Playground Series S5E3",
        input_dim=X.shape[1],
        n_targets=2,
        classify=True,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_obesity_classification_dataset() -> (
    tuple[Dataset, Dataset, Dataset, TaskConfig]
):
    """
    Loads the Obesity Classification dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    filenames = ["Obesity Classification.csv"]
    zip_name = "archive.zip"
    download_url = "https://www.kaggle.com/api/v1/datasets/download/sujithmandala/obesity-classification-dataset"
    _download_and_extract_kaggle_dataset(download_url, filenames, zip_name)

    data = pd.read_csv("Obesity Classification.csv")
    data = data.dropna()

    data["Label"] = LabelEncoder().fit_transform(data["Label"])
    data["Gender"] = LabelEncoder().fit_transform(data["Gender"])

    X = data.drop("Label", axis=1).values.astype(float)
    y = data["Label"].values.astype(int)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(X, y)

    config = TaskConfig(
        name="Obesity Classification",
        input_dim=X.shape[1],
        n_targets=4,
        classify=True,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_heart_failure_prediction_dataset() -> (
    tuple[Dataset, Dataset, Dataset, TaskConfig]
):
    """
    Loads the Heart Failure Prediction dataset for classification.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    # Download and extract the dataset if not already present
    filenames = ["heart.csv"]
    zip_name = "heart-failure-prediction.zip"
    download_url = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction"
    _download_and_extract_kaggle_dataset(download_url, filenames, zip_name)

    data = pd.read_csv(filenames[0])
    data = data.dropna()

    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
    data["ChestPainType"] = LabelEncoder().fit_transform(data["ChestPainType"])
    data["RestingECG"] = LabelEncoder().fit_transform(data["RestingECG"])
    data["ExerciseAngina"] = LabelEncoder().fit_transform(data["ExerciseAngina"])
    data["ST_Slope"] = LabelEncoder().fit_transform(data["ST_Slope"])

    X = data.drop("HeartDisease", axis=1).values.astype(float)
    y = data["HeartDisease"].values.astype(int)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(X, y)

    config = TaskConfig(
        name="Heart Failure Prediction",
        input_dim=X.shape[1],
        n_targets=2,
        classify=True,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_insurance_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the Insurance dataset for regression.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    # Download and extract the dataset if not already present
    filenames = ["insurance.csv"]
    zip_name = "insurance.zip"
    download_url = (
        "https://www.kaggle.com/api/v1/datasets/download/mirichoi0218/insurance"
    )
    _download_and_extract_kaggle_dataset(download_url, filenames, zip_name)

    data = pd.read_csv(filenames[0])
    data = data.dropna()

    data["sex"] = LabelEncoder().fit_transform(data["sex"])
    data["smoker"] = LabelEncoder().fit_transform(data["smoker"])
    data = pd.get_dummies(data, columns=["region"])

    X = data.drop("charges", axis=1).values.astype(float)
    y = data["charges"].values.astype(float)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(
        X, y, classify=False
    )

    config = TaskConfig(
        name="Insurance Dataset",
        input_dim=X.shape[1],
        n_targets=1,
        classify=False,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_boston_house_price_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the Boston House Price dataset for regression.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    # Download and extract the dataset if not already present
    filenames = ["boston.csv"]
    zip_name = "the-boston-houseprice-data.zip"
    download_url = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/the-boston-houseprice-data"
    _download_and_extract_kaggle_dataset(download_url, filenames, zip_name)

    data = pd.read_csv(filenames[0])
    data = data.dropna()

    X = data.drop("MEDV", axis=1).values.astype(float)
    y = data["MEDV"].values.astype(float)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(
        X, y, classify=False
    )

    config = TaskConfig(
        name="Boston House Price",
        input_dim=X.shape[1],
        n_targets=1,
        classify=False,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_concrete_compressive_strength_dataset() -> (
    tuple[Dataset, Dataset, Dataset, TaskConfig]
):
    """
    Loads the Concrete Compressive Strength dataset for regression.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    concrete_compressive_strength = fetch_ucirepo(id=165)
    X = concrete_compressive_strength.data.features
    y = concrete_compressive_strength.data.targets
    data = pd.DataFrame(X)
    data["concrete_compressive_strength"] = y
    X = data.drop("concrete_compressive_strength", axis=1).values.astype(float)
    y = data["concrete_compressive_strength"].values.astype(float)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(
        X, y, classify=False
    )

    config = TaskConfig(
        name="Concrete Compressive Strength",
        input_dim=X.shape[1],
        n_targets=1,
        classify=False,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config


def load_energy_efficiency_dataset() -> tuple[Dataset, Dataset, Dataset, TaskConfig]:
    """
    Loads the Energy Efficiency dataset for multi-target regression.

    Returns:
        tuple: A tuple containing the train, test, and validation datasets, and the task configuration.
    """
    energy_efficiency = fetch_ucirepo(id=181)
    X = energy_efficiency.data.features
    y = energy_efficiency.data.targets
    data = pd.DataFrame(X)
    X = data.values.astype(float)
    y = y.values.astype(float)

    (train_dataset, test_dataset), val_dataset = dataset_from_numpy(
        X, y, classify=False
    )

    config = TaskConfig(
        name="Energy Efficiency",
        input_dim=X.shape[1],
        n_targets=y.shape[1],
        classify=False,
        cnn=False,
        timeseries=False,
    )

    return train_dataset, val_dataset, test_dataset, config
