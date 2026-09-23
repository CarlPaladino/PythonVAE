import numpy as np


class Adam:
    def __init__(self, trainable_layers, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.trainable_layers = trainable_layers
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.step_number = 0
        self.weight_first_moments = []
        self.weight_second_moments = []
        self.bias_first_moments = []
        self.bias_second_moments = []
        for layer in trainable_layers:
            self.weight_first_moments.append(np.zeros_like(layer.weights))
            self.weight_second_moments.append(np.zeros_like(layer.weights))
            self.bias_first_moments.append(np.zeros_like(layer.biases))
            self.bias_second_moments.append(np.zeros_like(layer.biases))

    def update_matrix(self, parameters, gradients, first_moments, second_moments):
        first_moments *= self.beta_1
        first_moments += (1.0 - self.beta_1) * gradients
        squared_gradients = gradients * gradients
        second_moments *= self.beta_2
        second_moments += (1.0 - self.beta_2) * squared_gradients

        corrected_first = first_moments / self.first_correction
        corrected_second = second_moments / self.second_correction
        denominator = np.sqrt(corrected_second) + self.epsilon
        parameter_updates = self.learning_rate * corrected_first / denominator
        parameters -= parameter_updates

    def step(self):
        for layer in self.trainable_layers:
            if layer.dweights is None or layer.dbiases is None:
                raise RuntimeError("Every layer must run backward before Adam.step.")
        self.step_number += 1
        self.first_correction = 1.0 - self.beta_1**self.step_number
        self.second_correction = 1.0 - self.beta_2**self.step_number
        for index in range(len(self.trainable_layers)):
            layer = self.trainable_layers[index]
            self.update_matrix(layer.weights, layer.dweights, self.weight_first_moments[index], self.weight_second_moments[index])
            self.update_matrix(layer.biases, layer.dbiases, self.bias_first_moments[index], self.bias_second_moments[index])
