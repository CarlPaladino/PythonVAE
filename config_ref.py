TRAIN_PATH = "data/raw/emnist-balanced-train.csv"
TEST_PATH = "data/raw/emnist-balanced-test.csv"
CHECKPOINT_DIRECTORY = "checkpoints/ref"
GENERATED_IMAGE_PATH = "outputs/ref/generated_samples.svg"
LATENT_VISUALIZATION_PATH = "outputs/ref/latent_vector.svg"

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
RANDOM_SEED = 9

BATCH_SIZE = 256

CONV_CHANNELS = [8, 16]
HIDDEN_DIM = 256
LATENT_DIM = 24

KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

LEARNING_RATE = 0.001
KL_BETA = 1.0

SAMPLES = None
EPOCHS = 30

GENERATION_ROWS = 5
GENERATION_COLUMNS = 5
LATENT_SAMPLE_INDEX = 23


def _number_slug(value):
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def make_configuration_name(image_height, image_width, conv_channels, kernel_size, stride, padding, hidden_dim, latent_dim, kl_beta, learning_rate, batch_size, random_seed):
    channel_name = ""
    for channels in conv_channels:
        if channel_name:
            channel_name += "x"
        channel_name += str(channels)
    return f"conv-vae-v3_image-{image_height}x{image_width}_channels-{channel_name}_kernel-{kernel_size}_stride-{stride}_padding-{padding}_hidden-{hidden_dim}_latent-{latent_dim}_beta-{_number_slug(kl_beta)}_lr-{_number_slug(learning_rate)}_batch-{batch_size}_seed-{random_seed}"


CONFIGURATION_NAME = make_configuration_name(IMAGE_HEIGHT, IMAGE_WIDTH, CONV_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, HIDDEN_DIM, LATENT_DIM, KL_BETA, LEARNING_RATE, BATCH_SIZE, RANDOM_SEED)
CHECKPOINT_PATH = f"{CHECKPOINT_DIRECTORY}/{CONFIGURATION_NAME}/vae_weights.npz"
