from pathlib import Path
import numpy as np

import config
from src.data import load_sample, flatten_images, unflatten_image
from src.layers import Reparameterization
from src.model import VAE
from src.plots import text, image_elements, bar_elements, save_svg


def load_test_sample(sample_index):
    image, label = load_sample(config.TEST_PATH, sample_index,
                               config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    return image, flatten_images([image]), label


def main():
    checkpoint_path = Path(config.CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")
    original, inputs, label = load_test_sample(config.LATENT_SAMPLE_INDEX)
    model = VAE(config.INPUT_DIM, config.HIDDEN_DIMS, config.LATENT_DIM,
                np.random.default_rng(config.RANDOM_SEED))
    model.load_weights(checkpoint_path)
    mean, log_variance = model.encode(inputs)
    sampling = Reparameterization(np.random.default_rng(config.RANDOM_SEED))
    latent = sampling.forward(mean, log_variance)
    decoded = model.decode(latent)
    reconstruction = unflatten_image(decoded[0], config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    elements = [text(20, 25, f"Where a {model.latent_dim}-component latent vector comes from")]
    elements.append(text(20, 60, f"Input: {config.CLASSES[label]} (label {label})"))
    elements.extend(image_elements(original, 20, 80))
    elements.extend(bar_elements(mean[0], 220, 60, "Encoder mean"))
    elements.extend(bar_elements(sampling.standard_deviation[0], 580, 60, "Standard deviation"))
    elements.extend(bar_elements(sampling.epsilon[0], 20, 310, "Random noise"))
    elements.extend(bar_elements(latent[0], 380, 310, "Latent: mean + deviation * noise"))
    elements.append(text(740, 310, "Reconstruction"))
    elements.extend(image_elements(reconstruction, 740, 330))
    save_svg(config.LATENT_VISUALIZATION_PATH, 920, 550, elements)
    print(f"Open {config.LATENT_VISUALIZATION_PATH} in a browser.")


if __name__ == "__main__":
    main()
