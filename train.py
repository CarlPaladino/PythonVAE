import numpy as np

import config
from src.data import load_images, flatten_images
from src.losses import VAELoss
from src.model import VAE
from src.optimizer import Adam
from src.plots import image_grid


def load_data(path, limit=None):
    return load_images(path, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, limit)


def plot_samples(images, labels, rng, count=15):
    if count <= 0 or count > len(images):
        raise ValueError("Sample count must fit the dataset.")
    indices = list(range(len(images)))
    rng.shuffle(indices)
    selected_images = []
    selected_labels = []
    for position in range(count):
        index = indices[position]
        selected_images.append(images[index])
        selected_labels.append(f"Label {labels[index]}: {config.CLASSES[labels[index]]}")
    image_grid(selected_images, 1, count, "outputs/training_samples.svg",
               "Transformed EMNIST samples", selected_labels)


def train(model, images, rng, epochs, batch_size, learning_rate, beta, checkpoint_path):
    if epochs <= 0 or batch_size <= 0 or len(images) == 0:
        raise ValueError("Epochs, batch size, and sample count must be positive.")
    # Convert once at the boundary; batches remain NumPy arrays.
    images = np.asarray(images, dtype=model.dtype)
    loss_function = VAELoss(beta)
    optimizer = Adam(model.get_trainable_layers(), learning_rate=learning_rate)
    history = []
    for epoch in range(epochs):
        indices = rng.permutation(len(images))
        total_sum = 0.0
        reconstruction_sum = 0.0
        kl_sum = 0.0
        samples_processed = 0
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            # Select shuffled sample rows without slice shorthand.
            batch_positions = np.arange(start, end)
            batch_indices = np.take(indices, batch_positions)
            batch = np.take(images, batch_indices, axis=0)
            reconstruction, mean, log_variance = model.forward(batch)
            total_loss, reconstruction_loss, kl_loss = loss_function.forward(
                batch, reconstruction, mean, log_variance
            )
            reconstruction_gradients, mean_gradients, log_variance_gradients = loss_function.backward()
            model.backward(reconstruction_gradients, mean_gradients, log_variance_gradients)
            optimizer.step()
            count = len(batch)
            total_sum += total_loss * count
            reconstruction_sum += reconstruction_loss * count
            kl_sum += kl_loss * count
            samples_processed += count
            print(f"Epoch {epoch + 1}/{epochs}: {samples_processed}/{len(images)} samples", end="\r")
        averages = (total_sum / samples_processed,
                    reconstruction_sum / samples_processed,
                    kl_sum / samples_processed)
        history.append(averages)
        print(f"Epoch {epoch + 1}/{epochs}: loss={averages[0]:.4f}, reconstruction={averages[1]:.4f}, kl={averages[2]:.4f}")
        model.save_weights(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    return history


def main():
    rng = np.random.default_rng(config.RANDOM_SEED)
    print(f"Loading training data from {config.TRAIN_PATH}...", flush=True)
    train_images, _ = load_data(config.TRAIN_PATH, config.SAMPLES)
    print(f"Loaded {len(train_images)} samples. Initializing model...", flush=True)
    inputs = flatten_images(train_images)
    model = VAE(config.INPUT_DIM, config.HIDDEN_DIMS, config.LATENT_DIM, rng)
    print("Starting training...", flush=True)
    train(model, inputs, rng, config.EPOCHS, config.BATCH_SIZE,
          config.LEARNING_RATE, config.KL_BETA, config.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
