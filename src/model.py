from pathlib import Path
import json
import numpy as np

from src.layers import Dense, ReLU, Sigmoid, Reparameterization


class VAE:
    def __init__(self, input_dim, hidden_dims, latent_dim, rng, dtype=np.float32):
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one dimension.")
        self.dtype = np.dtype(dtype)
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim
        self.encoder_dense_layers = []
        self.encoder_activation_layers = []
        current_dimension = input_dim
        for hidden_dimension in hidden_dims:
            self.encoder_dense_layers.append(Dense(current_dimension, hidden_dimension, rng, self.dtype))
            self.encoder_activation_layers.append(ReLU())
            current_dimension = hidden_dimension
        self.mean_dense = Dense(current_dimension, latent_dim, rng, self.dtype)
        self.log_variance_dense = Dense(current_dimension, latent_dim, rng, self.dtype)
        self.reparameterization = Reparameterization(rng)
        self.decoder_dense_layers = []
        self.decoder_activation_layers = []
        current_dimension = latent_dim
        for index in range(len(hidden_dims) - 1, -1, -1):
            hidden_dimension = hidden_dims[index]
            self.decoder_dense_layers.append(Dense(current_dimension, hidden_dimension, rng, self.dtype))
            self.decoder_activation_layers.append(ReLU())
            current_dimension = hidden_dimension
        self.output_dense = Dense(current_dimension, input_dim, rng, self.dtype)
        self.output_sigmoid = Sigmoid()

    def encode(self, inputs):
        values = inputs
        for index in range(len(self.encoder_dense_layers)):
            values = self.encoder_dense_layers[index].forward(values)
            values = self.encoder_activation_layers[index].forward(values)
        mean = self.mean_dense.forward(values)
        log_variance = self.log_variance_dense.forward(values)
        return mean, log_variance

    def decode(self, latent_values):
        values = latent_values
        for index in range(len(self.decoder_dense_layers)):
            values = self.decoder_dense_layers[index].forward(values)
            values = self.decoder_activation_layers[index].forward(values)
        values = self.output_dense.forward(values)
        return self.output_sigmoid.forward(values)

    def forward(self, inputs):
        mean, log_variance = self.encode(inputs)
        latent_values = self.reparameterization.forward(mean, log_variance)
        reconstruction = self.decode(latent_values)
        return reconstruction, mean, log_variance

    def get_trainable_layers(self):
        layers = []
        for layer in self.encoder_dense_layers:
            layers.append(layer)
        layers.append(self.mean_dense)
        layers.append(self.log_variance_dense)
        for layer in self.decoder_dense_layers:
            layers.append(layer)
        layers.append(self.output_dense)
        return layers

    def save_weights(self, path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        layers = self.get_trainable_layers()
        if checkpoint_path.suffix == ".json":
            # Optional human-readable export; list conversion occurs only here.
            saved_layers = []
            for layer in layers:
                saved_layers.append({
                    "weights": layer.weights.tolist(),
                    "biases": layer.biases.tolist(),
                })
            checkpoint = {
                "format": "python-vae-v1",
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "latent_dim": self.latent_dim,
                "layers": saved_layers,
            }
            with checkpoint_path.open("w", encoding="utf-8") as output:
                json.dump(checkpoint, output, allow_nan=False)
        else:
            checkpoint = {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "latent_dim": self.latent_dim,
                "layer_count": len(layers),
            }
            for index in range(len(layers)):
                checkpoint[f"layer_{index}_weights"] = layers[index].weights
                checkpoint[f"layer_{index}_biases"] = layers[index].biases
            # Uncompressed arrays save faster than compressing every epoch.
            with checkpoint_path.open("wb") as output:
                np.savez(output, **checkpoint)

    def load_weights(self, path):
        checkpoint_path = Path(path)
        layers = self.get_trainable_layers()
        saved_weights = []
        saved_biases = []
        if checkpoint_path.suffix == ".json":
            with checkpoint_path.open(encoding="utf-8") as source:
                checkpoint = json.load(source)
            if checkpoint["format"] != "python-vae-v1":
                raise ValueError("Unknown checkpoint format.")
            saved_input_dim = checkpoint["input_dim"]
            saved_hidden_dims = checkpoint["hidden_dims"]
            saved_latent_dim = checkpoint["latent_dim"]
            for saved_layer in checkpoint["layers"]:
                saved_weights.append(np.asarray(saved_layer["weights"], dtype=self.dtype))
                saved_biases.append(np.asarray(saved_layer["biases"], dtype=self.dtype))
        else:
            # Also reads compressed .npz files from the original project.
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                saved_input_dim = int(checkpoint["input_dim"])
                saved_hidden_dims = checkpoint["hidden_dims"].tolist()
                saved_latent_dim = int(checkpoint["latent_dim"])
                for index in range(int(checkpoint["layer_count"])):
                    saved_weights.append(np.asarray(checkpoint[f"layer_{index}_weights"], dtype=self.dtype))
                    saved_biases.append(np.asarray(checkpoint[f"layer_{index}_biases"], dtype=self.dtype))
        if (
            saved_input_dim != self.input_dim
            or saved_hidden_dims != self.hidden_dims
            or saved_latent_dim != self.latent_dim
            or len(saved_weights) != len(layers)
        ):
            raise ValueError("Checkpoint architecture does not match.")
        # Check every layer before changing the model.
        for index in range(len(layers)):
            layer = layers[index]
            if saved_weights[index].shape != layer.weights.shape:
                raise ValueError("Checkpoint weight shape mismatch.")
            if saved_biases[index].shape != layer.biases.shape:
                raise ValueError("Checkpoint bias shape mismatch.")
            if not np.all(np.isfinite(saved_weights[index])):
                raise ValueError("Checkpoint weights must be finite.")
            if not np.all(np.isfinite(saved_biases[index])):
                raise ValueError("Checkpoint biases must be finite.")
        for index in range(len(layers)):
            np.copyto(layers[index].weights, saved_weights[index])
            np.copyto(layers[index].biases, saved_biases[index])

    def backward(self, reconstruction_gradients, mean_kl_gradients,
                 log_variance_kl_gradients):
        values = self.output_sigmoid.backward(reconstruction_gradients)
        values = self.output_dense.backward(values)
        for index in range(len(self.decoder_dense_layers) - 1, -1, -1):
            values = self.decoder_activation_layers[index].backward(values)
            values = self.decoder_dense_layers[index].backward(values)
        mean_reconstruction, log_variance_reconstruction = (
            self.reparameterization.backward(values)
        )
        mean_gradients = mean_reconstruction + mean_kl_gradients
        log_variance_gradients = log_variance_reconstruction + log_variance_kl_gradients
        mean_hidden = self.mean_dense.backward(mean_gradients)
        log_variance_hidden = self.log_variance_dense.backward(log_variance_gradients)
        # Both encoder branches contribute to the shared hidden representation.
        values = mean_hidden + log_variance_hidden
        for index in range(len(self.encoder_dense_layers) - 1, -1, -1):
            values = self.encoder_activation_layers[index].backward(values)
            values = self.encoder_dense_layers[index].backward(values)
        return values
