import numpy as np 

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true: np.ndarray
        The true labels
    y_pred: np.ndarray
        The predicted labels

    Returns
    -------
    rmse: float
        The root mean squared error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))