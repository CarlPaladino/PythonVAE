import pickle
from pathlib import Path


def save_checkpoint(path, encoder, decoder, rng, epoch, history, learning_rate, beta):
    checkpoint = {
        "encoder": encoder,
        "decoder": decoder,
        "rng_state": rng.bit_generator.state,
        "epoch": epoch,
        "history": history,
        "learning_rate": learning_rate,
        "beta": beta
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(checkpoint, file)

    temporary_path.replace(path)


def load_checkpoint(path):
    """Load a checkpoint created by this project from a trusted file."""
    with open(path, "rb") as file:
        return pickle.load(file)
