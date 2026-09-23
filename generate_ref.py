from pathlib import Path

import numpy as np

import config_ref
from src_ref.model import VAE
from src_ref.plots import image_grid


def generate_images(model, rng, sample_count):
    latent_shape = (sample_count, model.latent_dim)
    latent_values = rng.standard_normal(latent_shape, dtype=model.dtype)
    reconstructions = model.decode(latent_values)
    images = []
    for sample in range(sample_count):
        images.append(reconstructions[sample, 0])
    return images


def plot_images(images, rows, columns, output_path):
    image_grid(images, rows, columns, output_path, "Generated EMNIST samples")


def main():
    checkpoint_path = Path(config_ref.CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")
    rng = np.random.default_rng(config_ref.RANDOM_SEED)
    model = VAE(config_ref.IMAGE_HEIGHT, config_ref.IMAGE_WIDTH, config_ref.CONV_CHANNELS, config_ref.KERNEL_SIZE, config_ref.STRIDE, config_ref.PADDING, config_ref.HIDDEN_DIM, config_ref.LATENT_DIM, config_ref.KL_BETA, rng)
    model.load_weights(checkpoint_path)
    images = generate_images(model, rng, config_ref.GENERATION_ROWS * config_ref.GENERATION_COLUMNS)
    plot_images(images, config_ref.GENERATION_ROWS, config_ref.GENERATION_COLUMNS, config_ref.GENERATED_IMAGE_PATH)
    print(f"{config_ref.GENERATED_IMAGE_PATH} created.")


if __name__ == "__main__":
    main()
