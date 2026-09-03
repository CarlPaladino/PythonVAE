# NumPy VAE

A variational autoencoder implemented from scratch with NumPy and trained on
the EMNIST Balanced dataset.

Dataset source: https://www.kaggle.com/datasets/crawford/emnist

## Train

```powershell
.\.venv\Scripts\python.exe train.py
```

Each completed epoch saves the model weights in a configuration-specific
directory, for example:

```text
checkpoints/input-784_hidden-512x256_latent-16_beta-1_lr-0p001_batch-128_seed-9/vae_weights.npz
```

The input size, hidden-layer sizes, latent dimension, KL beta, learning rate,
batch size, and random seed are included automatically. Changing any of them
in `config.py` creates a separate checkpoint instead of overwriting another
configuration. `EPOCHS` is not part of the name, so increasing it keeps using
the checkpoint path for the same configuration.

## Generate samples

```powershell
.\.venv\Scripts\python.exe generate.py
```

This loads the saved checkpoint and writes a grid of generated characters to
`outputs/generated_samples.png`.

## Visualize one latent vector

```powershell
.\.venv\Scripts\python.exe visualize_latent.py
```

This shows how one test image produces the mean and standard deviation used to
sample a 16-component latent vector, then saves the figure to
`outputs/latent_vector.png`.
