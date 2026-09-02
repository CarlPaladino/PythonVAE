import numpy as np
import matplotlib.pyplot as plt

import config
from src.losses import VAELoss
from src.model import VAE
from src.optimizer import Adam

def load_data(path):
    raw = np.loadtxt(path, delimiter=",", dtype=np.uint8)
    labels = raw[:, 0].astype(np.int64)
    images = raw[:, 1:].reshape(-1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    images = images.transpose(0, 2, 1)
    images = images.astype(np.float32) / 255.0
    return images, labels

def plot_samples(images, labels, rng, count=15):
    indices = rng.choice(len(images), size=count, replace=False)

    figure, axes = plt.subplots(3, 5, figsize=(9, 6))

    for index, axis in zip(indices, axes.flat):
        label = labels[index]
        character = config.CLASSES[label]

        axis.imshow(images[index], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"Label {label}: {character}")
        axis.axis("off")

    figure.suptitle("Transformed EMNIST Samples")
    plt.tight_layout()
    plt.show()

def flatten_images(images):
    return images.reshape(len(images), config.INPUT_DIM)


def main():
    rng = np.random.default_rng(config.RANDOM_SEED)

    train_images, _ = load_data(config.TRAIN_PATH)
    x_train = flatten_images(train_images)

    model = VAE(
        config.INPUT_DIM,
        config.HIDDEN_DIMS,
        config.LATENT_DIM,
        rng,
    )

    loss_function = VAELoss(beta=config.KL_BETA)
    optimizer = Adam(
        model.get_trainable_layers(),
        learning_rate=config.LEARNING_RATE,
    )

    for epoch in range(config.EPOCHS):
        shuffled_indices = rng.permutation(len(x_train))

        epoch_total_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_kl_loss = 0.0
        samples_processed = 0

        print(f"Epoch {epoch + 1}/{config.EPOCHS}")

        for start in range(0, len(x_train), config.BATCH_SIZE):
            end = start + config.BATCH_SIZE
            batch_indices = shuffled_indices[start:end]
            x_batch = x_train[batch_indices]

            reconstruction, mean, log_variance = model.forward(x_batch)

            total_loss, reconstruction_loss, kl_loss = loss_function.forward(
                x_batch,
                reconstruction,
                mean,
                log_variance,
            )

            (
                reconstruction_gradients,
                mean_kl_gradients,
                log_variance_kl_gradients,
            ) = loss_function.backward()

            model.backward(
                reconstruction_gradients,
                mean_kl_gradients,
                log_variance_kl_gradients,
            )

            optimizer.step()

            batch_size = len(x_batch)
            epoch_total_loss = epoch_total_loss + total_loss * batch_size
            epoch_reconstruction_loss = (
                epoch_reconstruction_loss + reconstruction_loss * batch_size
            )
            epoch_kl_loss = epoch_kl_loss + kl_loss * batch_size
            samples_processed = samples_processed + batch_size

            print(
                f"Processed {samples_processed}/{len(x_train)} samples - "
                f"loss: {total_loss:.4f} - "
                f"reconstruction: {reconstruction_loss:.4f} - "
                f"kl: {kl_loss:.4f}",
                end="\r",
            )

        average_total_loss = epoch_total_loss / samples_processed
        average_reconstruction_loss = (
            epoch_reconstruction_loss / samples_processed
        )
        average_kl_loss = epoch_kl_loss / samples_processed

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} - "
            f"loss: {average_total_loss:.4f} - "
            f"reconstruction: {average_reconstruction_loss:.4f} - "
            f"kl: {average_kl_loss:.4f}"
        )

        model.save_weights(config.CHECKPOINT_PATH)
        print(f"Saved checkpoint to {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
    
