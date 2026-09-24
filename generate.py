"""Generate characters and visualize a saved VAE without training it.

Run: python generate.py
Open plot windows too: python generate.py --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from proposals.vae_speedup.vae import decode, encode
from src.checkpoints import load_checkpoint
from src.data import load_images


def latest_checkpoint():
    paths = list(Path("checkpoints").glob("*/epoch-*.pkl"))
    if not paths:
        raise FileNotFoundError("No epoch checkpoints found. Train the model first or pass --checkpoint.")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def generate_characters(decoder, count, rng):
    latent_size = decoder["weights"].shape[0]
    latent_vectors = rng.normal(0.0, 1.0, size=(count, latent_size))
    images = []

    for index in range(count):
        image, _ = decode(latent_vectors[index], decoder)
        images.append(image)

    return np.array(images), latent_vectors


def encode_images(inputs, encoder):
    latent_size = len(encoder["mean_biases"])
    means = np.empty((len(inputs), latent_size))
    standard_deviations = np.empty(means.shape)

    for index in range(len(inputs)):
        mean, log_variance, _ = encode(inputs[index], encoder)
        means[index] = mean
        standard_deviations[index] = np.exp(0.5 * log_variance)

    return means, standard_deviations


def project_latents(means, generated_latents, latent_axes=None):
    latent_size = means.shape[1]

    if latent_axes is not None:
        coordinates = np.take(means, latent_axes, axis=1)
        generated_coordinates = np.take(generated_latents, latent_axes, axis=1)
        axis_labels = (f"Latent dimension {latent_axes[0]}", f"Latent dimension {latent_axes[1]}")
        method = "Selected latent dimensions"
    elif latent_size == 1:
        coordinates = np.zeros((len(means), 2))
        generated_coordinates = np.zeros((len(generated_latents), 2))
        for index in range(len(means)):
            coordinates[index, 0] = means[index, 0]
        for index in range(len(generated_latents)):
            generated_coordinates[index, 0] = generated_latents[index, 0]
        axis_labels = ("Latent dimension 0", "Unused axis (one-dimensional model)")
        method = "One-dimensional latent space"
    elif latent_size == 2:
        coordinates = means.copy()
        generated_coordinates = generated_latents.copy()
        axis_labels = ("Latent dimension 0", "Latent dimension 1")
        method = "Two-dimensional latent space"
    else:
        # Fit PCA to the encoded image means, then use the SAME projection
        # for random latent samples. A projection loses information.
        center = np.mean(means, axis=0)
        centered = means - center
        _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
        basis = np.take(directions, [0, 1], axis=0).T
        coordinates = np.matmul(centered, basis)
        generated_coordinates = np.matmul(generated_latents - center, basis)

        variance = singular_values * singular_values
        total_variance = np.sum(variance)
        percentages = np.zeros(2)
        if total_variance > 0:
            percentages = 100.0 * np.take(variance, [0, 1]) / total_variance
        axis_labels = (f"PC1 ({percentages[0]:.1f}% of encoded-mean variance)", f"PC2 ({percentages[1]:.1f}% of encoded-mean variance)")
        method = f"PCA projection of {latent_size} latent dimensions"

    return coordinates, generated_coordinates, axis_labels, method


def save_figure(figure, output_directory, name):
    figure.savefig(output_directory / f"{name}.png", dpi=160)
    figure.savefig(output_directory / f"{name}.svg")


def plot_generated(images, output_directory, title, columns=5):
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)

    for index in range(rows * columns):
        axis = axes[index // columns, index % columns]
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_title(f"Sample {index + 1}", fontsize=10)

    figure.suptitle(f"Random characters: z sampled from N(0, I)\n{title}")
    figure.tight_layout()
    save_figure(figure, output_directory, "generated_characters")
    return figure


def plot_latent_space(means, labels, generated_latents, output_directory, title, latent_axes=None):
    coordinates, generated_coordinates, axis_labels, method = project_latents(means, generated_latents, latent_axes)
    figure, axis = plt.subplots(figsize=(10, 8))
    points = axis.scatter(np.take(coordinates, 0, axis=1), np.take(coordinates, 1, axis=1), c=labels, cmap=plt.get_cmap("turbo", 47), vmin=-0.5, vmax=46.5, s=20, alpha=0.75, label="Encoded image means")
    colorbar = figure.colorbar(points, ax=axis)
    colorbar.set_label("EMNIST balanced class label (0-46)")
    colorbar.set_ticks([0, 10, 20, 30, 40, 46])

    axis.scatter(np.take(generated_coordinates, 0, axis=1), np.take(generated_coordinates, 1, axis=1), marker="x", c="black", s=55, label="Random latent samples (generated grid)")
    for index in range(len(generated_coordinates)):
        axis.annotate(str(index + 1), (generated_coordinates[index, 0], generated_coordinates[index, 1]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    axis.set_xlabel(axis_labels[0])
    axis.set_ylabel(axis_labels[1])
    axis.set_title(f"{method}\n{title}")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    save_figure(figure, output_directory, "latent_space")
    return figure


def plot_latent_statistics(means, standard_deviations, output_directory, title):
    dimensions = np.arange(means.shape[1])
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].errorbar(dimensions, np.mean(means, axis=0), yerr=np.std(means, axis=0), fmt="o", capsize=4)
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_ylabel("Encoder mean")
    axes[0].set_title("Average mean; error bars show spread between images")

    axes[1].bar(dimensions, np.mean(standard_deviations, axis=0))
    axes[1].axhline(1, color="gray", linestyle="--", label="Standard-normal prior")
    axes[1].set_ylabel("Average predicted std. deviation")
    axes[1].set_xlabel("Latent dimension (zero-based)")
    axes[1].set_xticks(dimensions)
    axes[1].legend()

    figure.suptitle(f"Latent-variable statistics\n{title}")
    figure.tight_layout()
    save_figure(figure, output_directory, "latent_statistics")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the most recently saved epoch checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data/raw/emnist-balanced-test.csv"))
    parser.add_argument("--map-samples", type=int, default=500, help="Number of CSV images to encode for the plots.")
    parser.add_argument("--count", type=int, default=25, help="Number of randomly generated images.")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--latent-axes", type=int, nargs=2, metavar=("X", "Y"), help="Plot two zero-based dimensions instead of PCA.")
    parser.add_argument("--output", type=Path, help="Optional output directory.")
    parser.add_argument("--show", action="store_true", help="Also open interactive plot windows.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or latest_checkpoint()
    checkpoint = load_checkpoint(checkpoint_path)
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    output_directory = args.output or (Path("outputs") / checkpoint_path.parent.name / f"{checkpoint_path.stem}-generation-seed-{args.seed}")
    output_directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    generated_images, generated_latents = generate_characters(decoder, args.count, rng)
    height, width = generated_images[0, 0].shape
    inputs, labels = load_images(args.data, height=height, width=width, limit=args.map_samples)
    means, standard_deviations = encode_images(inputs, encoder)

    title = f"{checkpoint_path.parent.name} / epoch {checkpoint['epoch']}"
    figures = [plot_generated(generated_images, output_directory, title), plot_latent_space(means, labels, generated_latents, output_directory, f"{title}; {len(inputs)} images from {args.data.name}", args.latent_axes), plot_latent_statistics(means, standard_deviations, output_directory, title)]

    # Keep the exact numbers behind the figures for later analysis.
    np.savez(output_directory / "latent_values.npz", means=means, standard_deviations=standard_deviations, labels=labels, generated_latents=generated_latents, generated_images=generated_images, checkpoint=str(checkpoint_path), data_path=str(args.data), seed=args.seed)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved generated characters, latent-space plot, and latent statistics to: {output_directory}")
    print("Black crosses in latent_space match the numbered generated characters.")
    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
