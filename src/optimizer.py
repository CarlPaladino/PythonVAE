import numpy as np


class Adam:
    def __init__(
        self,
        trainable_layers,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    ):
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

        for layer in self.trainable_layers:
            self.weight_first_moments.append(np.zeros_like(layer.weights))
            self.weight_second_moments.append(np.zeros_like(layer.weights))
            self.bias_first_moments.append(np.zeros_like(layer.biases))
            self.bias_second_moments.append(np.zeros_like(layer.biases))

    def step(self):
        for layer in self.trainable_layers:
            if layer.dweights is None or layer.dbiases is None:
                raise RuntimeError(
                    "Every trainable layer must run backward() before Adam.step()."
                )

        self.step_number = self.step_number + 1

        for index in range(len(self.trainable_layers)):
            layer = self.trainable_layers[index]

            weight_first_moment = self.weight_first_moments[index]
            weight_second_moment = self.weight_second_moments[index]
            bias_first_moment = self.bias_first_moments[index]
            bias_second_moment = self.bias_second_moments[index]

            weight_first_moment = (
                self.beta_1 * weight_first_moment
                + (1.0 - self.beta_1) * layer.dweights
            )
            weight_second_moment = (
                self.beta_2 * weight_second_moment
                + (1.0 - self.beta_2) * layer.dweights * layer.dweights
            )
            bias_first_moment = (
                self.beta_1 * bias_first_moment
                + (1.0 - self.beta_1) * layer.dbiases
            )
            bias_second_moment = (
                self.beta_2 * bias_second_moment
                + (1.0 - self.beta_2) * layer.dbiases * layer.dbiases
            )

            self.weight_first_moments[index] = weight_first_moment
            self.weight_second_moments[index] = weight_second_moment
            self.bias_first_moments[index] = bias_first_moment
            self.bias_second_moments[index] = bias_second_moment

            first_moment_correction = 1.0 - self.beta_1 ** self.step_number
            second_moment_correction = 1.0 - self.beta_2 ** self.step_number

            corrected_weight_first_moment = (
                weight_first_moment / first_moment_correction
            )
            corrected_weight_second_moment = (
                weight_second_moment / second_moment_correction
            )
            corrected_bias_first_moment = (
                bias_first_moment / first_moment_correction
            )
            corrected_bias_second_moment = (
                bias_second_moment / second_moment_correction
            )

            weight_update = corrected_weight_first_moment / (
                np.sqrt(corrected_weight_second_moment) + self.epsilon
            )
            bias_update = corrected_bias_first_moment / (
                np.sqrt(corrected_bias_second_moment) + self.epsilon
            )

            layer.weights = layer.weights - self.learning_rate * weight_update
            layer.biases = layer.biases - self.learning_rate * bias_update
