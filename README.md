# NumPy VAE

A variational autoencoder implemented from scratch with NumPy and trained on
the EMNIST Balanced dataset.

Dataset source: https://www.kaggle.com/datasets/crawford/emnist

## Train

```powershell
.\.venv\Scripts\python.exe train.py
```

Each completed epoch saves the model weights to
`checkpoints/vae_weights.npz`.

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
