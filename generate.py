from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from src.model import VAE


def generate_images(model, rng, sample_count):
    latent_values = rng.standard_normal(
        (sample_count, config.LATENT_DIM)
    )
    latent_values = latent_values.astype(np.float32)

    flattened_images = model.decode(latent_values)
    images = flattened_images.reshape(
        sample_count,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
    )

    return images


def plot_images(images, rows, columns, output_path, show_window=True):
    expected_image_count = rows * columns

    if len(images) != expected_image_count:
        raise ValueError(
            "The number of images must equal rows multiplied by columns."
        )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2, rows * 2),
    )
    axes = np.asarray(axes).reshape(rows, columns)

    for row in range(rows):
        for column in range(columns):
            image_index = row * columns + column
            axis = axes[row, column]

            axis.imshow(
                images[image_index],
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axis.axis("off")

    figure.suptitle("Generated EMNIST Samples")
    figure.tight_layout()

    image_path = Path(output_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_path, dpi=150)

    if show_window:
        plt.show()

    plt.close(figure)


def main():
    checkpoint_path = Path(config.CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run train.py first."
        )

    rng = np.random.default_rng(config.RANDOM_SEED)

    model = VAE(
        config.INPUT_DIM,
        config.HIDDEN_DIMS,
        config.LATENT_DIM,
        rng,
    )
    model.load_weights(checkpoint_path)

    sample_count = config.GENERATION_ROWS * config.GENERATION_COLUMNS
    images = generate_images(model, rng, sample_count)

    plot_images(
        images,
        config.GENERATION_ROWS,
        config.GENERATION_COLUMNS,
        config.GENERATED_IMAGE_PATH,
    )

    print(f"Saved generated samples to {config.GENERATED_IMAGE_PATH}")


if __name__ == "__main__":
    main()
