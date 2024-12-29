import numpy as np
from si.metrics.rmse import rmse

class KNNRegressor:
    def __init__(self, k: int = 3):
        """
        K-Nearest Neighbors Regressor.

        Parameters
        ----------
        k: int
            The number of neighbors to consider
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """
        Fit the model.

        Parameters
        ----------
        X: np.ndarray
            The training data
        y: np.ndarray
            The target values

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values.

        Parameters
        ----------
        X: np.ndarray
            The data to predict

        Returns
        -------
        y_pred: np.ndarray
            The predicted target values
        """
        y_pred = []
        for x in X:
            # Compute distances to all training samples
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get indices of k nearest neighbors
            neighbors_idx = np.argsort(distances)[:self.k]
            # Compute the mean of the neighbors' target values
            neighbors_mean = np.mean(self.y_train[neighbors_idx])
            y_pred.append(neighbors_mean)
        return np.array(y_pred)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model using RMSE.

        Parameters
        ----------
        X: np.ndarray
            The data to evaluate
        y: np.ndarray
            The true target values

        Returns
        -------
        rmse: float
            The root mean squared error
        """
        y_pred = self.predict(X)
        return rmse(y, y_pred)
