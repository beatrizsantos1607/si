import numpy as np
from si.neural_networks.activation import ActivationLayer


class TanhActivation(ActivationLayer):
    """
    Tanh activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Tanh  activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        numerador=np.exp(input)-np.exp(-input) #seguir formula
        denominador=np.exp(input) + np.exp(-input)
        return numerador/denominador

    def derivative(self, input: np.ndarray):
        """
        Derivative of the Tanh  activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        tanh_output = self.activation_function(input)
        return 1 - (tanh_output ** 2)
    
if __name__ == '__main__':
    input_data = np.array([[1.0, -2.0, 3.0]])
    
    
    tanh_activation = TanhActivation()
    tanh_output = tanh_activation.forward_propagation(input_data, training=True)
    tanh_derivative = tanh_activation.derivative(input_data)

    print("\nTanh Activation Output:")
    print(tanh_output)
    print("Tanh Activation Derivative:")
    print(tanh_derivative)