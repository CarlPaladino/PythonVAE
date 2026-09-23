import json
from pathlib import Path

import numpy as np

from src_ref.layers import Conv2D, Dense, ReLU, Reparameterization, Sigmoid, Upsample2D


class VAE:
    def __init__(self, image_height, image_width, conv_channels, kernel_size, stride, padding, hidden_dim, latent_dim, beta, rng):
        self.dtype = np.float32
        self.image_height = image_height
        self.image_width = image_width
        self.conv_channels = list(conv_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder_convs = []
        self.encoder_relus = []
        current_channels = 1
        feature_height = image_height
        feature_width = image_width
        for channels in self.conv_channels:
            self.encoder_convs.append(Conv2D(current_channels, channels, kernel_size, rng, stride=stride, padding=padding))
            self.encoder_relus.append(ReLU())
            feature_height = (feature_height + 2 * padding - kernel_size) // stride + 1
            feature_width = (feature_width + 2 * padding - kernel_size) // stride + 1
            current_channels = channels

        self.feature_shape = (current_channels, feature_height, feature_width)
        self.feature_dim = current_channels * feature_height * feature_width
        self.encoder_dense = Dense(self.feature_dim, hidden_dim, rng)
        self.encoder_dense_relu = ReLU()
        self.mean_dense = Dense(hidden_dim, latent_dim, rng)
        self.log_variance_dense = Dense(hidden_dim, latent_dim, rng)
        self.reparameterization = Reparameterization(rng)

        self.decoder_hidden = Dense(latent_dim, hidden_dim, rng)
        self.decoder_hidden_relu = ReLU()
        self.decoder_features = Dense(hidden_dim, self.feature_dim, rng)
        self.decoder_features_relu = ReLU()
        self.decoder_upsamples = []
        self.decoder_convs = []
        self.decoder_relus = []
        for index in range(len(self.conv_channels) - 1, -1, -1):
            output_channels = 1
            if index > 0:
                output_channels = self.conv_channels[index - 1]
            self.decoder_upsamples.append(Upsample2D())
            self.decoder_convs.append(Conv2D(current_channels, output_channels, kernel_size, rng, padding=padding))
            if index > 0:
                self.decoder_relus.append(ReLU())
            current_channels = output_channels
        self.output_sigmoid = Sigmoid()

    def compute_loss(self, targets, reconstruction, mean, log_variance):
        batch_size = len(targets)
        variance = np.exp(log_variance)

        predictions = np.clip(reconstruction, 1e-7, 1.0 - 1e-7)
        unclipped_predictions = (reconstruction > 1e-7) & (reconstruction < 1.0 - 1e-7)

        pixel_losses = -(targets * np.log(predictions) + (1.0 - targets) * np.log(1.0 - predictions))
        reconstruction_loss = np.mean(np.sum(pixel_losses, axis=(1, 2, 3)))

        kl_components = 0.5 * (mean * mean + variance - 1.0 - log_variance)
        kl_loss = np.mean(np.sum(kl_components, axis=1))
        total_loss = reconstruction_loss + self.beta * kl_loss

        reconstruction_gradients = (1.0 - targets) / (1.0 - predictions) - targets / predictions
        reconstruction_gradients *= unclipped_predictions
        reconstruction_gradients /= batch_size

        mean_gradients = self.beta * mean / batch_size
        log_variance_gradients = 0.5 * self.beta * (variance - 1.0) / batch_size

        if not np.isfinite(total_loss):
            raise FloatingPointError("VAE loss became nonfinite. Check inputs and reduce the learning rate.")
        losses = (float(total_loss), float(reconstruction_loss), float(kl_loss))
        gradients = (reconstruction_gradients, mean_gradients, log_variance_gradients)
        return losses, gradients

    def encode(self, inputs):
        values = np.asarray(inputs, dtype=self.dtype)
        expected_shape = (len(values), 1, self.image_height, self.image_width)
        if values.shape != expected_shape or len(values) == 0:
            raise ValueError("Expected a nonempty batch shaped (samples, 1, image_height, image_width).")
        for index in range(len(self.encoder_convs)):
            values = self.encoder_convs[index].forward(values)
            values = self.encoder_relus[index].forward(values)
        values = values.reshape(len(values), self.feature_dim)
        values = self.encoder_dense.forward(values)
        values = self.encoder_dense_relu.forward(values)
        mean = self.mean_dense.forward(values)
        log_variance = self.log_variance_dense.forward(values)
        return mean, log_variance

    def decode(self, latent_values):
        values = np.asarray(latent_values, dtype=self.dtype)
        values = self.decoder_hidden.forward(values)
        values = self.decoder_hidden_relu.forward(values)
        values = self.decoder_features.forward(values)
        values = self.decoder_features_relu.forward(values)
        feature_channels, feature_height, feature_width = self.feature_shape
        values = values.reshape(len(values), feature_channels, feature_height, feature_width)
        for index in range(len(self.decoder_convs)):
            values = self.decoder_upsamples[index].forward(values)
            values = self.decoder_convs[index].forward(values)
            if index < len(self.decoder_relus):
                values = self.decoder_relus[index].forward(values)
        return self.output_sigmoid.forward(values)

    def forward(self, inputs):
        mean, log_variance = self.encode(inputs)
        latent_values = self.reparameterization.forward(mean, log_variance)
        reconstruction = self.decode(latent_values)

        return reconstruction, mean, log_variance

    def get_trainable_layers(self):
        layers = []
        for layer in self.encoder_convs:
            layers.append(layer)
        layers.append(self.encoder_dense)
        layers.append(self.mean_dense)
        layers.append(self.log_variance_dense)
        layers.append(self.decoder_hidden)
        layers.append(self.decoder_features)
        for layer in self.decoder_convs:
            layers.append(layer)
        return layers

    def parameter_count(self):
        count = 0
        for layer in self.get_trainable_layers():
            count += layer.weights.size + layer.biases.size
        return count

    def architecture(self):
        return {"format": "python-conv-vae-v3", "image_height": self.image_height, "image_width": self.image_width, "conv_channels": self.conv_channels, "kernel_size": self.kernel_size, "stride": self.stride, "padding": self.padding, "hidden_dim": self.hidden_dim, "latent_dim": self.latent_dim}

    def save_weights(self, path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.architecture()
        layers = self.get_trainable_layers()
        checkpoint["layer_count"] = len(layers)
        for index in range(len(layers)):
            layer = layers[index]
            checkpoint[f"layer_{index}_weights"] = layer.weights
            checkpoint[f"layer_{index}_biases"] = layer.biases
        if checkpoint_path.suffix == ".json":
            json_checkpoint = {}
            for key in checkpoint:
                value = checkpoint[key]
                if isinstance(value, np.ndarray):
                    json_checkpoint[key] = value.tolist()
                else:
                    json_checkpoint[key] = value
            checkpoint = json_checkpoint
            checkpoint_path.write_text(json.dumps(checkpoint, allow_nan=False), encoding="utf-8")
        else:
            with checkpoint_path.open("wb") as output:
                np.savez(output, **checkpoint)

    def load_weights(self, path):
        checkpoint_path = Path(path)
        if checkpoint_path.suffix == ".json":
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        else:
            with np.load(checkpoint_path, allow_pickle=False) as saved:
                checkpoint = {}
                for key in saved.files:
                    checkpoint[key] = saved[key]
        expected_architecture = self.architecture()
        for key in expected_architecture:
            value = expected_architecture[key]
            if key not in checkpoint or not np.array_equal(checkpoint[key], value):
                raise ValueError("Checkpoint architecture does not match this convolutional VAE.")
        layers = self.get_trainable_layers()
        if checkpoint.get("layer_count") != len(layers):
            raise ValueError("Checkpoint layer count does not match.")
        parameters = []
        for index in range(len(layers)):
            layer = layers[index]
            for name in ("weights", "biases"):
                target = getattr(layer, name)
                source = np.asarray(checkpoint[f"layer_{index}_{name}"], dtype=self.dtype)
                if source.shape != target.shape or not np.all(np.isfinite(source)):
                    raise ValueError("Checkpoint contains invalid parameter shapes or values.")
                parameters.append((target, source))
        for target, source in parameters:
            np.copyto(target, source)

    def backward(self, reconstruction_gradients, mean_kl_gradients, log_variance_kl_gradients):
        values = self.output_sigmoid.backward(reconstruction_gradients)
        for index in range(len(self.decoder_convs) - 1, -1, -1):
            if index < len(self.decoder_relus):
                values = self.decoder_relus[index].backward(values)
            values = self.decoder_convs[index].backward(values)
            values = self.decoder_upsamples[index].backward(values)
        values = values.reshape(len(values), self.feature_dim)
        values = self.decoder_features_relu.backward(values)
        values = self.decoder_features.backward(values)
        values = self.decoder_hidden_relu.backward(values)
        values = self.decoder_hidden.backward(values)

        mean_reconstruction, log_variance_reconstruction = self.reparameterization.backward(values)
        mean_hidden = self.mean_dense.backward(mean_reconstruction + mean_kl_gradients)
        log_variance_hidden = self.log_variance_dense.backward(log_variance_reconstruction + log_variance_kl_gradients)
        values = mean_hidden + log_variance_hidden
        values = self.encoder_dense_relu.backward(values)
        values = self.encoder_dense.backward(values)
        feature_channels, feature_height, feature_width = self.feature_shape
        values = values.reshape(len(values), feature_channels, feature_height, feature_width)
        for index in range(len(self.encoder_convs) - 1, -1, -1):
            values = self.encoder_relus[index].backward(values)
            values = self.encoder_convs[index].backward(values)
        return values
