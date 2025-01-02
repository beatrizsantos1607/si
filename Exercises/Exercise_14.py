import numpy as np
from si.neural_networks.losses import LossFunction

class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross Entropy loss function.
    Measures the dissimilarity between predicted class probabilities and true one-hot encoded class labels
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15) #limitar valores do y_pred, para nao andar com valores de infinito e assim
        return -np.sum(y_true * np.log(p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p)
    
if __name__ == '__main__':

    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]])  
    
    cce_loss = CategoricalCrossEntropy()
    cce_result = cce_loss.loss(y_true, y_pred)
    cce_derivative = cce_loss.derivative(y_true, y_pred)

    print(f"Categorical Cross Entropy Loss: {cce_result}")
    print(f"Categorical Cross Entropy Derivative: {cce_derivative}") 