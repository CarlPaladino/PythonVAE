from pathlib import Path
import numpy as np

import config
from src.data import unflatten_image
from src.model import VAE
from src.plots import image_grid


def generate_images(model, rng, sample_count):
    latent_shape = (sample_count, model.latent_dim)
    latent_values = rng.standard_normal(latent_shape, dtype=model.dtype)
    flattened_images = model.decode(latent_values)
    images = []
    for pixels in flattened_images:
        images.append(unflatten_image(pixels, config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    return images


def plot_images(images, rows, columns, output_path):
    image_grid(images, rows, columns, output_path, "Generated EMNIST samples")


def main():
    checkpoint_path = Path(config.CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")
    rng = np.random.default_rng(config.RANDOM_SEED)
    model = VAE(config.INPUT_DIM, config.HIDDEN_DIMS, config.LATENT_DIM, rng)
    model.load_weights(checkpoint_path)
    images = generate_images(model, rng, config.GENERATION_ROWS * config.GENERATION_COLUMNS)
    plot_images(images, config.GENERATION_ROWS, config.GENERATION_COLUMNS, config.GENERATED_IMAGE_PATH)
    print(f"Open {config.GENERATED_IMAGE_PATH} in a browser.")


if __name__ == "__main__":
    main()
