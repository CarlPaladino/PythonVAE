import numpy as np
from numpy.typing import NDArray

class Dense:
    def __init__(self, input_dim, output_dim, rng):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inputs = None
        self.dweights = None
        self.dbiases = None

        weight_scale = np.sqrt(2.0 / input_dim)
        self.weights = rng.normal(loc=0.0, scale=weight_scale, size=(input_dim, output_dim)).astype(np.float32)
        self.biases = np.zeros((1, output_dim), dtype=np.float32)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        return np.dot(dvalues, self.weights.T)

class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        dinputs = dvalues.copy()
        dinputs[self.inputs <= 0] = 0
        return dinputs

class Sigmoid:
    def __init__(self):
        self.output = None
        
    def forward(self, inputs):
        self.output = np.empty_like(inputs,dtype=np.result_type(inputs.dtype, np.float32))

        positive = inputs >= 0
        self.output[positive] = 1 / (1 + np.exp(-inputs[positive]))

        exp_inputs = np.exp(inputs[~positive])
        self.output[~positive] = exp_inputs / (1 + exp_inputs)

        return self.output

    def backward(self, dvalues):
        return dvalues * self.output * (1 - self.output)


class Reparameterization:
    def __init__(self, rng):
        self.rng = rng
        self.epsilon = None
        self.standard_deviation = None

    def forward(self, mean, log_variance):
        if mean.shape != log_variance.shape:
            raise ValueError("mean and log_variance must have the same shape.")

        mean_values = np.asarray(mean, dtype=np.float32)
        log_variance_values = np.asarray(log_variance, dtype=np.float32)

        self.standard_deviation = np.exp(0.5 * log_variance_values)
        self.epsilon = self.rng.standard_normal(mean_values.shape)
        self.epsilon = self.epsilon.astype(np.float32)

        scaled_noise = self.standard_deviation * self.epsilon
        latent_values = mean_values + scaled_noise
        return latent_values

    def backward(self, latent_gradients):
        gradient_values = np.asarray(latent_gradients, dtype=np.float32)

        mean_gradients = gradient_values.copy()
        log_variance_gradients = gradient_values * self.epsilon * 0.5 * self.standard_deviation

        return mean_gradients, log_variance_gradients
