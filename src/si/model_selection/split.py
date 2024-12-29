from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Perform stratified splitting of the dataset into training and testing sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # Set random state
    np.random.seed(random_state)

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(dataset.y, return_counts=True)
    train_indices = []
    test_indices = []

    # Loop through each unique class
    for cls, count in zip(unique_classes, class_counts):
        # Get indices of the current class
        cls_indices = np.where(dataset.y == cls)[0]
        # Shuffle indices
        np.random.shuffle(cls_indices)
        # Calculate the number of test samples for the current class
        n_test = int(count * test_size)
        # Split into test and train indices
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    # Shuffle the final train and test indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Create the train and test datasets
    train = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return train, test
