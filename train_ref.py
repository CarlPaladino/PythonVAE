import numpy as np

import config_ref
from src_ref.data import load_images
from src_ref.model import VAE
from src_ref.optimizer import Adam
from src_ref.plots import image_grid


def load_data(path, limit=None):
    return load_images(path, config_ref.IMAGE_HEIGHT, config_ref.IMAGE_WIDTH, limit)


def plot_samples(images, labels, rng, count=15):
    indices = list(range(len(images)))
    rng.shuffle(indices)
    selected_images = []
    selected_labels = []

    for position in range(count):
        index = indices[position]
        selected_images.append(images[index])
        selected_labels.append(f"Label {labels[index]}: {config_ref.CLASSES[labels[index]]}")

    image_grid(selected_images, 1, count, "outputs/training_samples.svg", "Transformed EMNIST samples", selected_labels)


def train(images, rng, epochs, learning_rate, beta, checkpoint_path):
    model = VAE(config_ref.IMAGE_HEIGHT, config_ref.IMAGE_WIDTH, config_ref.CONV_CHANNELS, config_ref.KERNEL_SIZE, config_ref.STRIDE, config_ref.PADDING, config_ref.HIDDEN_DIM, config_ref.LATENT_DIM, beta, rng)
    images = np.asarray(images, dtype=model.dtype)
    optimizer = Adam(model.get_trainable_layers(), learning_rate=learning_rate)
    history = []

    for epoch in range(epochs):
        total_sum = 0.0
        reconstruction_sum = 0.0
        kl_sum = 0.0
        samples_processed = 0
        indices = rng.permutation(len(images))

        for start in range(0, len(images), config_ref.BATCH_SIZE):
            end = min(start + config_ref.BATCH_SIZE, len(images))
            batch_positions = np.arange(start, end)
            batch_indices = np.take(indices, batch_positions)
            batch = np.take(images, batch_indices, axis=0)

            reconstruction, mean, log_variance = model.forward(batch)
            losses, gradients = model.compute_loss(batch, reconstruction, mean, log_variance)
            total_loss, reconstruction_loss, kl_loss = losses
            reconstruction_gradients, mean_gradients, log_variance_gradients = gradients
            model.backward(reconstruction_gradients, mean_gradients, log_variance_gradients)
            optimizer.step()
            count = len(batch)
            total_sum += total_loss * count
            reconstruction_sum += reconstruction_loss * count
            kl_sum += kl_loss * count
            samples_processed += count
            print(f"Epoch {epoch + 1}/{epochs}: {samples_processed}/{len(images)} samples", end="\r")

        averages = (total_sum / samples_processed, reconstruction_sum / samples_processed, kl_sum / samples_processed)
        history.append(averages)
        print(f"Epoch {epoch + 1}/{epochs}: loss={averages[0]:.4f}, reconstruction={averages[1]:.4f}, kl={averages[2]:.4f}")
        model.save_weights(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    return history


def main():
    rng = np.random.default_rng(config_ref.RANDOM_SEED)
    train_images, _ = load_data(config_ref.TRAIN_PATH, config_ref.SAMPLES)
    sample_count, height, width = train_images.shape
    inputs = train_images.reshape(sample_count, 1, height, width)

    train(inputs, rng, config_ref.EPOCHS, config_ref.LEARNING_RATE, config_ref.KL_BETA, config_ref.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
