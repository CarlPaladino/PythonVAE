TRAIN_PATH = "data/raw/emnist-balanced-train.csv"
TEST_PATH = "data/raw/emnist-balanced-test.csv"
CHECKPOINT_DIRECTORY = "checkpoints"
GENERATED_IMAGE_PATH = "outputs/generated_samples.png"
LATENT_VISUALIZATION_PATH = "outputs/latent_vector.png"

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
RANDOM_SEED = 9

INPUT_DIM = IMAGE_HEIGHT * IMAGE_WIDTH

BATCH_SIZE = 128
HIDDEN_DIMS = [600, 400, 200]
LATENT_DIM = 24

LEARNING_RATE = 0.001
KL_BETA = 1.0

EPOCHS = 100

GENERATION_ROWS = 5
GENERATION_COLUMNS = 5
LATENT_SAMPLE_INDEX = 0


def _number_slug(value):
    """Format a number so it can safely be used in a file name."""
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def make_configuration_name(
    input_dim,
    hidden_dims,
    latent_dim,
    kl_beta,
    learning_rate,
    batch_size,
    random_seed,
):
    """Return a readable, stable name for a training configuration."""
    hidden_layers = "x".join(str(dimension) for dimension in hidden_dims)

    return (
        f"input-{input_dim}"
        f"_hidden-{hidden_layers}"
        f"_latent-{latent_dim}"
        f"_beta-{_number_slug(kl_beta)}"
        f"_lr-{_number_slug(learning_rate)}"
        f"_batch-{batch_size}"
        f"_seed-{random_seed}"
    )


# EPOCHS is intentionally omitted: increasing the training duration should keep
# saving to the checkpoint for the same experiment configuration.
CONFIGURATION_NAME = make_configuration_name(
    INPUT_DIM,
    HIDDEN_DIMS,
    LATENT_DIM,
    KL_BETA,
    LEARNING_RATE,
    BATCH_SIZE,
    RANDOM_SEED,
)
CHECKPOINT_PATH = (
    f"{CHECKPOINT_DIRECTORY}/{CONFIGURATION_NAME}/vae_weights.npz"
)
