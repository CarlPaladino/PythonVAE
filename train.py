def train():
    epochs = 15
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")


def main():
    rng = np.random.default_rng(config.RANDOM_SEED)
    train_images, _ = load_data(config.TRAIN_PATH, config.SAMPLES)
    sample_count, height, width = train_images.shape
    inputs = train_images.reshape(sample_count, 1, height, width)

    train(inputs, rng, config.EPOCHS, config.LEARNING_RATE, config.KL_BETA, config.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
