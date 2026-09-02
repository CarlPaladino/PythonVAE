from pathlib import Path

import numpy as np

from src.layers import Dense, ReLU, Sigmoid, Reparameterization

class VAE:
    def __init__(self, input_dim, hidden_dims, latent_dim, rng):
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one dimension.")

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim

        self.encoder_dense_layers = []
        self.encoder_activation_layers = []

        current_dimension = input_dim

        for index in range(len(hidden_dims)):
            hidden_dimension = hidden_dims[index]

            dense_layer = Dense(
                current_dimension,
                hidden_dimension,
                rng,
            )
            activation_layer = ReLU()

            self.encoder_dense_layers.append(dense_layer)
            self.encoder_activation_layers.append(activation_layer)

            current_dimension = hidden_dimension

        self.mean_dense = Dense(
            current_dimension,
            latent_dim,
            rng,
        )
        self.log_variance_dense = Dense(
            current_dimension,
            latent_dim,
            rng,
        )

        self.reparameterization = Reparameterization(rng)

        self.decoder_dense_layers = []
        self.decoder_activation_layers = []

        current_dimension = latent_dim

        for index in range(len(hidden_dims) - 1, -1, -1):
            hidden_dimension = hidden_dims[index]

            dense_layer = Dense(current_dimension, hidden_dimension, rng)
            activation_layer = ReLU()

            self.decoder_dense_layers.append(dense_layer)
            self.decoder_activation_layers.append(activation_layer)

            current_dimension = hidden_dimension

        self.output_dense = Dense(current_dimension, input_dim, rng)
        self.output_sigmoid = Sigmoid()

    def encode(self, inputs):
        values = inputs

        for index in range(len(self.encoder_dense_layers)):
            dense_layer = self.encoder_dense_layers[index]
            activation_layer = self.encoder_activation_layers[index]

            values = dense_layer.forward(values)
            values = activation_layer.forward(values)

        mean = self.mean_dense.forward(values)
        log_variance = self.log_variance_dense.forward(values)

        return mean, log_variance

    def decode(self, latent_values):
        values = latent_values

        for index in range(len(self.decoder_dense_layers)):
            dense_layer = self.decoder_dense_layers[index]
            activation_layer = self.decoder_activation_layers[index]

            values = dense_layer.forward(values)
            values = activation_layer.forward(values)

        values = self.output_dense.forward(values)
        reconstruction = self.output_sigmoid.forward(values)

        return reconstruction

    def forward(self, inputs):
        mean, log_variance = self.encode(inputs)

        latent_values = self.reparameterization.forward(
            mean,
            log_variance,
        )

        reconstruction = self.decode(latent_values)

        return reconstruction, mean, log_variance

    def get_trainable_layers(self):
        trainable_layers = []

        for layer in self.encoder_dense_layers:
            trainable_layers.append(layer)

        trainable_layers.append(self.mean_dense)
        trainable_layers.append(self.log_variance_dense)

        for layer in self.decoder_dense_layers:
            trainable_layers.append(layer)

        trainable_layers.append(self.output_dense)
        return trainable_layers

    def save_weights(self, path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        trainable_layers = self.get_trainable_layers()

        checkpoint_values = {
            "input_dim": np.array(self.input_dim, dtype=np.int64),
            "hidden_dims": np.array(self.hidden_dims, dtype=np.int64),
            "latent_dim": np.array(self.latent_dim, dtype=np.int64),
            "layer_count": np.array(len(trainable_layers), dtype=np.int64),
        }

        for index in range(len(trainable_layers)):
            layer = trainable_layers[index]
            checkpoint_values[f"layer_{index}_weights"] = layer.weights
            checkpoint_values[f"layer_{index}_biases"] = layer.biases

        np.savez_compressed(checkpoint_path, **checkpoint_values)

    def load_weights(self, path):
        checkpoint_path = Path(path)
        trainable_layers = self.get_trainable_layers()

        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            saved_input_dim = int(checkpoint["input_dim"])
            saved_hidden_dims = checkpoint["hidden_dims"].tolist()
            saved_latent_dim = int(checkpoint["latent_dim"])
            saved_layer_count = int(checkpoint["layer_count"])

            architecture_matches = (
                saved_input_dim == self.input_dim
                and saved_hidden_dims == self.hidden_dims
                and saved_latent_dim == self.latent_dim
                and saved_layer_count == len(trainable_layers)
            )

            if not architecture_matches:
                raise ValueError(
                    "The checkpoint architecture does not match this VAE."
                )

            for index in range(len(trainable_layers)):
                layer = trainable_layers[index]
                saved_weights = checkpoint[f"layer_{index}_weights"]
                saved_biases = checkpoint[f"layer_{index}_biases"]

                if saved_weights.shape != layer.weights.shape:
                    raise ValueError(
                        f"Weight shape does not match for trainable layer {index}."
                    )

                if saved_biases.shape != layer.biases.shape:
                    raise ValueError(
                        f"Bias shape does not match for trainable layer {index}."
                    )

                layer.weights[:] = saved_weights
                layer.biases[:] = saved_biases

    def backward(
        self,
        reconstruction_gradients,
        mean_kl_gradients,
        log_variance_kl_gradients,
    ):
        values = self.output_sigmoid.backward(reconstruction_gradients)
        values = self.output_dense.backward(values)

        for index in range(len(self.decoder_dense_layers) - 1, -1, -1):
            activation_layer = self.decoder_activation_layers[index]
            dense_layer = self.decoder_dense_layers[index]

            values = activation_layer.backward(values)
            values = dense_layer.backward(values)

        latent_gradients = values

        mean_reconstruction_gradients, log_variance_reconstruction_gradients = (
            self.reparameterization.backward(latent_gradients)
        )

        mean_gradients = (
            mean_reconstruction_gradients + mean_kl_gradients
        )
        log_variance_gradients = (
            log_variance_reconstruction_gradients
            + log_variance_kl_gradients
        )

        mean_hidden_gradients = self.mean_dense.backward(mean_gradients)
        log_variance_hidden_gradients = self.log_variance_dense.backward(
            log_variance_gradients
        )

        values = mean_hidden_gradients + log_variance_hidden_gradients

        for index in range(len(self.encoder_dense_layers) - 1, -1, -1):
            activation_layer = self.encoder_activation_layers[index]
            dense_layer = self.encoder_dense_layers[index]

            values = activation_layer.backward(values)
            values = dense_layer.backward(values)

        input_gradients = values
        return input_gradients
