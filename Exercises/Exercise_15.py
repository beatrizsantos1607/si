import numpy as np

class Adam:
    """
    Adam Optimizer
    Combines the benefits of RMSprop and SGD with momentum.

    Parameters:
    - learning_rate: The learning rate for updating weights.
    - beta_1: Exponential decay rate for the 1st moment estimates (default 0.9).
    - beta_2: Exponential decay rate for the 2nd moment estimates (default 0.999).
    - epsilon: A small constant for numerical stability (default 1e-8).
    """

    def __init__(self, learning_rate= 0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        
        super().__init__(learning_rate) #vai buscar o parametro learning rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0  # Time step (epoch)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update weights using Adam optimization algorithm.

        Parameters:
        - w: Current weights.
        - grad_loss_w: Gradient of the loss function with respect to the weights.

        Returns:
        - Updated weights.
        """
        if self.m is None or self.v is None:
            # Initialize m and v as zeros matrices with the same shape as weights
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        else:
            self.t += 1

            # Update biased first moment estimate
            self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

            # Update biased second raw moment estimate
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.beta_1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v / (1 - self.beta_2 ** self.t)

            # Update weights
            w_updated = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return w_updated

if __name__ == '__main__':
    w = np.array([2.0, 1.0])
    grad_loss_w = np.array([1.0, -2.0])
    learning_rates = [0.01, 0.03, 0.05]
    for lr in learning_rates:
        adam_optimizer = Adam(learning_rate=lr)

        updated_w_adam = adam_optimizer.update(w, grad_loss_w)
        print(f"Updated Weights (Adam, LR={lr}):", updated_w_adam)