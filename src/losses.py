import numpy as np


class VAELoss:
    def __init__(self, beta=1.0):
        self.beta = beta
        self.targets = None
        self.reconstruction = None
        self.mean = None
        self.log_variance = None
        self.batch_size = None

    def forward(self, targets, reconstruction, mean, log_variance):
        if targets.shape != reconstruction.shape:
            raise ValueError(
                "targets and reconstruction must have the same shape."
            )

        if mean.shape != log_variance.shape:
            raise ValueError(
                "mean and log_variance must have the same shape."
            )

        if targets.shape[0] != mean.shape[0]:
            raise ValueError(
                "The reconstruction and latent values must use the same batch size."
            )

        if targets.shape[0] == 0:
            raise ValueError("The batch must contain at least one sample.")

        self.targets = np.asarray(targets, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.log_variance = np.asarray(log_variance, dtype=np.float32)
        self.batch_size = self.targets.shape[0]

        reconstruction_values = np.asarray(reconstruction, dtype=np.float32)
        self.reconstruction = np.clip(reconstruction_values, 1e-7, 1.0 - 1e-7)

        positive_pixel_loss = self.targets * np.log(self.reconstruction)
        negative_pixel_loss = (1.0 - self.targets) * np.log(1.0 - self.reconstruction)
        pixel_loss = -(positive_pixel_loss + negative_pixel_loss)
        reconstruction_loss_per_sample = np.sum(pixel_loss, axis=1)

        variance = np.exp(self.log_variance)
        kl_loss_per_dimension = (1.0 + self.log_variance - self.mean * self.mean - variance)
        kl_loss_per_sample = -0.5 * np.sum(kl_loss_per_dimension, axis=1)

        reconstruction_loss = np.mean(reconstruction_loss_per_sample)
        kl_loss = np.mean(kl_loss_per_sample)
        total_loss = reconstruction_loss + self.beta * kl_loss

        return float(total_loss), float(reconstruction_loss), float(kl_loss)

    def backward(self):
        if self.targets is None:
            raise RuntimeError("VAELoss.forward() must be called before backward().")

        reconstruction_gradients = ((1.0 - self.targets) / (1.0 - self.reconstruction) - self.targets / self.reconstruction)
        reconstruction_gradients = reconstruction_gradients / self.batch_size

        mean_gradients = self.beta * self.mean
        mean_gradients = mean_gradients / self.batch_size

        variance = np.exp(self.log_variance)
        log_variance_gradients = 0.5 * self.beta * (variance - 1.0)
        log_variance_gradients = log_variance_gradients / self.batch_size

        return (reconstruction_gradients, mean_gradients, log_variance_gradients)
