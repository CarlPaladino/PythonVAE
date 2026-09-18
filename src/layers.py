import numpy as np


class Dense:
    def __init__(self, input_dim, output_dim, rng, dtype=np.float32):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Layer dimensions must be positive.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inputs = None
        self.dweights = None
        self.dbiases = None

        weight_scale = np.sqrt(2.0 / input_dim)
        self.weights = rng.normal(0.0, weight_scale, (input_dim, output_dim)).astype(dtype)
        self.biases = np.zeros((1, output_dim), dtype=dtype)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("Expected inputs shaped (samples, input_dim).")
        self.inputs = inputs

        weighted_inputs = np.matmul(inputs, self.weights)
        outputs = weighted_inputs + self.biases
        return outputs

    def backward(self, output_gradients):
        if self.inputs is None:
            raise RuntimeError("Call forward before backward.")
        expected_shape = (self.inputs.shape[0], self.output_dim)
        if output_gradients.shape != expected_shape:
            raise ValueError("Dense gradient shape mismatch.")

        inputs_by_feature = self.inputs.T
        self.dweights = np.matmul(inputs_by_feature, output_gradients)
        self.dbiases = np.sum(output_gradients, axis=0, keepdims=True)

        weights_by_neuron = self.weights.T
        input_gradients = np.matmul(output_gradients, weights_by_neuron)
        return input_gradients


class ReLU:
    def __init__(self):
        self.positive_inputs = None

    def forward(self, inputs):
        self.positive_inputs = inputs > 0.0
        outputs = np.maximum(inputs, 0.0)
        return outputs

    def backward(self, output_gradients):
        if self.positive_inputs is None:
            raise RuntimeError("Call forward before backward.")
        if output_gradients.shape != self.positive_inputs.shape:
            raise ValueError("ReLU gradient shape mismatch.")
        input_gradients = output_gradients * self.positive_inputs
        return input_gradients


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        negative_magnitude = -np.abs(inputs)
        exponential = np.exp(negative_magnitude)
        denominator = 1.0 + exponential
        positive_result = 1.0 / denominator
        negative_result = exponential / denominator
        self.output = np.where(inputs >= 0.0, positive_result, negative_result)
        return self.output

    def backward(self, output_gradients):
        if self.output is None:
            raise RuntimeError("Call forward before backward.")
        if output_gradients.shape != self.output.shape:
            raise ValueError("Sigmoid gradient shape mismatch.")
        sigmoid_derivative = self.output * (1.0 - self.output)
        input_gradients = output_gradients * sigmoid_derivative
        return input_gradients


class Reparameterization:
    def __init__(self, rng):
        self.rng = rng
        self.epsilon = None
        self.standard_deviation = None

    def forward(self, mean, log_variance):
        if mean.shape != log_variance.shape:
            raise ValueError("Mean and log variance shapes must match.")
        log_standard_deviation = 0.5 * log_variance
        self.standard_deviation = np.exp(log_standard_deviation)
        self.epsilon = self.rng.standard_normal(mean.shape, dtype=mean.dtype)
        scaled_noise = self.standard_deviation * self.epsilon
        latent_values = mean + scaled_noise
        return latent_values

    def backward(self, latent_gradients):
        if self.epsilon is None:
            raise RuntimeError("Call forward before backward.")
        if latent_gradients.shape != self.epsilon.shape:
            raise ValueError("Latent gradient shape mismatch.")
        mean_gradients = latent_gradients.copy()
        deviation_gradients = latent_gradients * self.epsilon
        log_variance_gradients = deviation_gradients * 0.5 * self.standard_deviation
        return mean_gradients, log_variance_gradients
