from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from src.model import VAE


def load_test_sample(sample_index):
    if sample_index < 0:
        raise ValueError("sample_index cannot be negative.")

    row = np.loadtxt(
        config.TEST_PATH,
        delimiter=",",
        dtype=np.uint8,
        skiprows=sample_index,
        max_rows=1,
    )

    if row.size == 0:
        raise IndexError("sample_index is outside the test dataset.")

    label = int(row[0])
    image = row[1:].reshape(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    image = image.T
    image = image.astype(np.float32) / 255.0

    flattened_image = image.reshape(1, config.INPUT_DIM)
    return image, flattened_image, label


def plot_component_values(axis, values, title, color):
    component_numbers = np.arange(1, config.LATENT_DIM + 1)

    axis.bar(component_numbers, values, color=color)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Latent component")
    axis.set_xticks(component_numbers)
    axis.grid(axis="y", alpha=0.25)


def main():
    checkpoint_path = Path(config.CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run train.py first."
        )

    original_image, flattened_image, label = load_test_sample(
        config.LATENT_SAMPLE_INDEX
    )

    model_rng = np.random.default_rng(config.RANDOM_SEED)
    model = VAE(
        config.INPUT_DIM,
        config.HIDDEN_DIMS,
        config.LATENT_DIM,
        model_rng,
    )
    model.load_weights(checkpoint_path)

    mean, log_variance = model.encode(flattened_image)
    standard_deviation = np.exp(0.5 * log_variance)

    sampling_rng = np.random.default_rng(config.RANDOM_SEED)
    epsilon = sampling_rng.standard_normal(mean.shape).astype(np.float32)
    latent_values = mean + standard_deviation * epsilon

    flattened_reconstruction = model.decode(latent_values)
    reconstructed_image = flattened_reconstruction.reshape(
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
    )

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))

    character = config.CLASSES[label]
    axes[0, 0].imshow(original_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(f"Input image: label {label} ({character})")
    axes[0, 0].axis("off")

    plot_component_values(
        axes[0, 1],
        mean[0],
        "Encoder output: mean (mu)",
        "tab:blue",
    )
    plot_component_values(
        axes[0, 2],
        standard_deviation[0],
        "Encoder output: standard deviation",
        "tab:orange",
    )
    plot_component_values(
        axes[1, 0],
        epsilon[0],
        "Random noise: epsilon ~ N(0, 1)",
        "tab:gray",
    )
    plot_component_values(
        axes[1, 1],
        latent_values[0],
        "Latent vector: z = mu + sigma * epsilon",
        "tab:purple",
    )

    axes[1, 2].imshow(
        reconstructed_image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 2].set_title("Decoder output: reconstruction")
    axes[1, 2].axis("off")

    figure.suptitle(
        "Where one 16-component latent vector comes from",
        fontsize=16,
    )
    figure.tight_layout()

    output_path = Path(config.LATENT_VISUALIZATION_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"Saved latent visualization to {output_path}")


if __name__ == "__main__":
    main()
