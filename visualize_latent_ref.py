from pathlib import Path

import numpy as np

import config_ref
from src_ref.data import load_sample
from src_ref.layers import Reparameterization
from src_ref.model import VAE
from src_ref.plots import bar_elements, image_elements, save_svg, text


def load_test_sample(sample_index):
    image, label = load_sample(config_ref.TEST_PATH, sample_index, config_ref.IMAGE_HEIGHT, config_ref.IMAGE_WIDTH)
    height, width = image.shape
    inputs = image.reshape(1, 1, height, width)
    return image, inputs, label


def main():
    checkpoint_path = Path(config_ref.CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")
    original, inputs, label = load_test_sample(config_ref.LATENT_SAMPLE_INDEX)
    model = VAE(config_ref.IMAGE_HEIGHT, config_ref.IMAGE_WIDTH, config_ref.CONV_CHANNELS, config_ref.KERNEL_SIZE, config_ref.STRIDE, config_ref.PADDING, config_ref.HIDDEN_DIM, config_ref.LATENT_DIM, config_ref.KL_BETA, np.random.default_rng(config_ref.RANDOM_SEED))
    model.load_weights(checkpoint_path)
    mean, log_variance = model.encode(inputs)
    sampling = Reparameterization(np.random.default_rng(config_ref.RANDOM_SEED))
    latent = sampling.forward(mean, log_variance)
    decoded = model.decode(latent)
    reconstruction = decoded[0, 0]
    elements = [text(20, 25, f"Where a {model.latent_dim}-component latent vector comes from")]
    elements.append(text(20, 60, f"Input: {config_ref.CLASSES[label]} (label {label})"))
    elements.extend(image_elements(original, 20, 80))
    elements.extend(bar_elements(mean[0], 220, 60, "Encoder mean"))
    elements.extend(bar_elements(sampling.standard_deviation[0], 580, 60, "Standard deviation"))
    elements.extend(bar_elements(sampling.epsilon[0], 20, 310, "Random noise"))
    elements.extend(bar_elements(latent[0], 380, 310, "Latent: mean + deviation * noise"))
    elements.append(text(740, 310, "Reconstruction"))
    elements.extend(image_elements(reconstruction, 740, 330))
    save_svg(config_ref.LATENT_VISUALIZATION_PATH, 920, 550, elements)
    print(f"{config_ref.LATENT_VISUALIZATION_PATH} created.")


if __name__ == "__main__":
    main()
