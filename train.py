from datetime import datetime
from pathlib import Path

import numpy as np

from proposals.vae_speedup.plots import save_reconstructions
from proposals.vae_speedup.vae import initialize_decoder, initialize_encoder, train_step
from src.checkpoints import save_checkpoint
from src.data import load_images

# from src.plots import save_reconstructions
# from src.vae import initialize_decoder, initialize_encoder, train_step


def train(inputs, encoder, decoder, rng, epochs, learning_rate, beta):
    sample_count = len(inputs)
    history = []
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    checkpoint_directory = Path("checkpoints") / run_name
    output_directory = Path("outputs") / run_name

    print(f"Checkpoints: {checkpoint_directory}")
    print(f"Reconstruction images: {output_directory}")

    for epoch in range(epochs):
        sample_order = rng.permutation(sample_count)

        total_loss_sum = 0.0
        pixel_loss_sum = 0.0
        latent_loss_sum = 0.0

        sample_count = 0

        for sample_index in sample_order:
            print(f"\rSample {sample_count}/{len(inputs)}", end="", flush=True)
            sample_count += 1
            total_loss, pixel_loss, latent_loss = train_step(inputs[sample_index], encoder, decoder, rng, learning_rate, beta)

            total_loss_sum += total_loss
            pixel_loss_sum += pixel_loss
            latent_loss_sum += latent_loss

        print()

        average_total = total_loss_sum / sample_count
        average_pixel = pixel_loss_sum / sample_count
        average_latent = latent_loss_sum / sample_count

        history.append({"total_loss": average_total, "reconstruction_loss": average_pixel, "kl_loss": average_latent})

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {average_total:.4f} | Reconstruction: {average_pixel:.4f} | KL: {average_latent:.4f}")

        epoch_name = f"epoch-{epoch + 1:03d}"
        save_checkpoint(checkpoint_directory / f"{epoch_name}.pkl", encoder, decoder, rng, epoch + 1, history, learning_rate, beta)
        save_reconstructions(output_directory / f"{epoch_name}.svg", inputs, encoder, decoder, epoch + 1)

    return history


def main():
    random_seed = 9
    sample_limit = None
    epochs = 50
    learning_rate = 0.001
    beta = 1.0

    filter_counts = [8, 16]
    latent_size = 16

    rng = np.random.default_rng(random_seed)

    inputs, _ = load_images("data/raw/emnist-balanced-train.csv", limit=sample_limit)
    image_shape = inputs[0].shape

    encoder = initialize_encoder(image_shape, filter_counts, latent_size, rng, kernel_size=3, stride=2, padding=1)
    decoder = initialize_decoder(encoder, image_channels=image_shape[0], latent_size=latent_size, rng=rng)

    history = train(inputs, encoder, decoder, rng, epochs, learning_rate, beta)

    return encoder, decoder, history


if __name__ == "__main__":
    main()
