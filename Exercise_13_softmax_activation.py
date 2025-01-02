import numpy as np
from si.neural_networks.activation import ActivationLayer

class Softmaxactivation(ActivationLayer):
    """
    Softmax activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        shifted_input = input - np.max(input, axis=-1, keepdims=True)
        #fico com um novo input penso que seja para evitar altos valores exponenciais 
        # Compute the exponentials of the shifted input
        exp_input = np.exp(shifted_input)

        # Compute the softmax output
        softmax_output = exp_input / np.sum(exp_input, axis=-1, keepdims=True) #formula

        return softmax_output

    def derivative(self, input: np.ndarray):
        """
        Derivative of the Softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input)*(1-self.activation_function(input))

if __name__ == '__main__':
    input_data = np.array([[1.0, -2.0, 3.0]])
    
    softmax_activation =Softmaxactivation()
    softmax_output = softmax_activation.forward_propagation(input_data, training=True)
    softmax_derivative = softmax_activation.derivative(input_data)

    print("\nSoftmax Activation Output:")
    print(softmax_output)
    print("Softmax Activation Derivative:")
    print(softmax_derivative)