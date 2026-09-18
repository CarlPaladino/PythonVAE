import numpy as np


class VAELoss:
    def __init__(self, beta=1.0):
        self.beta = beta
        self.targets = None

    def forward(self, targets, reconstruction, mean, log_variance):
        if targets.ndim != 2 or targets.shape != reconstruction.shape:
            raise ValueError("Targets and reconstruction must have matching matrix shapes.")
        if mean.ndim != 2 or mean.shape != log_variance.shape:
            raise ValueError("Mean and log variance must have matching matrix shapes.")
        if targets.shape[0] == 0 or targets.shape[0] != mean.shape[0]:
            raise ValueError("Expected matching, nonempty batches.")

        self.targets = targets
        self.mean = mean
        self.batch_size = targets.shape[0]
        self.variance = np.exp(log_variance)
        self.reconstruction = np.clip(reconstruction, 1e-7, 1.0 - 1e-7)
        above_lower_bound = reconstruction > 1e-7
        below_upper_bound = reconstruction < 1.0 - 1e-7
        self.unclipped_predictions = above_lower_bound & below_upper_bound

        positive_pixel_loss = targets * np.log(self.reconstruction)
        negative_pixel_loss = (1.0 - targets) * np.log(1.0 - self.reconstruction)
        pixel_losses = -(positive_pixel_loss + negative_pixel_loss)
        reconstruction_per_sample = np.sum(pixel_losses, axis=1)
        reconstruction_loss = np.mean(reconstruction_per_sample)

        squared_mean = mean * mean
        kl_per_component = 0.5 * (squared_mean + self.variance - 1.0 - log_variance)
        kl_per_sample = np.sum(kl_per_component, axis=1)
        kl_loss = np.mean(kl_per_sample)
        total_loss = reconstruction_loss + self.beta * kl_loss
        return float(total_loss), float(reconstruction_loss), float(kl_loss)

    def backward(self):
        if self.targets is None:
            raise RuntimeError("Call forward before backward.")

        negative_pixel_gradient = (1.0 - self.targets) / (1.0 - self.reconstruction)
        positive_pixel_gradient = self.targets / self.reconstruction
        reconstruction_gradients = negative_pixel_gradient - positive_pixel_gradient
        reconstruction_gradients *= self.unclipped_predictions
        reconstruction_gradients /= self.batch_size

        mean_gradients = self.beta * self.mean / self.batch_size
        log_variance_gradients = 0.5 * self.beta * (self.variance - 1.0) / self.batch_size
        return reconstruction_gradients, mean_gradients, log_variance_gradients
