from functools import lru_cache

import numpy as np


def convolve_image(image, kernel, bias=0.0, stride=1, padding=0):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = convolution_output_size(height, kernel_height, stride, padding)
    output_width = convolution_output_size(width, kernel_width, stride, padding)

    padded_image = pad_image(image, padding)
    output = np.zeros((output_height, output_width), dtype=float)

    for output_row in range(output_height):
        for output_column in range(output_width):
            start_row = output_row * stride
            start_column = output_column * stride

            output[output_row, output_column] = apply_kernel(padded_image, kernel, start_row, start_column, bias)

    return output


def pad_image(image, padding):
    return np.pad(image, ((padding, padding), (padding, padding)))



def apply_kernel(image, kernel, start_row, start_column, bias=0.0):
    kernel_height, kernel_width = kernel.shape
    total = 0.0

    for kernel_row in range(kernel_height):
        for kernel_column in range(kernel_width):
            image_row = start_row + kernel_row
            image_column = start_column + kernel_column

            pixel = image[image_row, image_column]
            weight = kernel[kernel_row, kernel_column]

            total += pixel * weight

    return total + bias


def convolution_output_size(input_size, kernel_size, stride=1, padding=0):
    padded_size = input_size + 2 * padding
    available_movement = padded_size - kernel_size
    output_size = available_movement // stride + 1

    return output_size


def relu(feature_map):
    return np.maximum(feature_map, 0)



def flatten(feature_maps):
    return feature_maps.reshape(-1).copy()



def dense(inputs, weights, biases):
    return np.matmul(inputs, weights) + biases



def initialize_dense(input_count, neuron_count, rng):
    scale = np.sqrt(1.0 / input_count)

    weights = rng.normal(loc=0.0, scale=scale, size=(input_count, neuron_count))

    biases = np.zeros(neuron_count)

    return weights, biases


def initialize_convolution(input_channels, filter_count, kernel_size, rng):
    inputs_per_filter = input_channels * kernel_size * kernel_size
    scale = np.sqrt(2.0 / inputs_per_filter)  # He initialization for convolutional layers

    kernels = rng.normal(loc=0.0, scale=scale, size=(filter_count, input_channels, kernel_size, kernel_size))

    biases = np.zeros(filter_count)

    return kernels, biases


def convolution(inputs, kernels, biases, stride=1, padding=0, cache=None):
    channels, height, width = inputs.shape
    filter_count, _, kernel_height, kernel_width = kernels.shape
    output_height = convolution_output_size(height, kernel_height, stride, padding)
    output_width = convolution_output_size(width, kernel_width, stride, padding)
    patches = image_to_patches(inputs, kernel_height, kernel_width, stride, padding)
    filter_weights = kernels_to_matrix(kernels)
    weighted_sums = np.matmul(patches, filter_weights)
    weighted_sums += biases
    if cache is not None:
        cache["patches"] = patches
    return weighted_sums.T.reshape(filter_count, output_height, output_width)



def activate_feature_maps(feature_maps):
    return np.maximum(feature_maps, 0)



def sample_latent(mean, log_variance, rng):
    noise = rng.normal(0.0, 1.0, size=len(mean))
    latent = mean + np.exp(0.5 * log_variance) * noise
    return latent, noise



def unflatten(values, channels, height, width):
    return values.reshape(channels, height, width).copy()



def upsample(feature_maps, scale=2):
    return np.repeat(np.repeat(feature_maps, scale, axis=1), scale, axis=2)



def sigmoid(feature_maps):
    exponential = np.exp(-np.abs(feature_maps))
    return np.where(
        feature_maps >= 0,
        1.0 / (1.0 + exponential),
        exponential / (1.0 + exponential)
    )



def reconstruction_loss(original, reconstruction):
    difference = reconstruction - original
    return 0.5 * np.sum(difference * difference)



def kl_loss(mean, log_variance):
    return 0.5 * np.sum(mean * mean + np.exp(log_variance) - 1.0 - log_variance)



def reconstruction_loss_backward(original, reconstruction):
    return reconstruction - original



def sigmoid_backward(sigmoid_output, output_gradients):
    return output_gradients * sigmoid_output * (1.0 - sigmoid_output)



def apply_kernel_backward(image, kernel, start_row, start_column, output_gradient):
    image_gradients = np.zeros(image.shape, dtype=float)
    kernel_gradients = np.empty(kernel.shape, dtype=float)

    kernel_height, kernel_width = kernel.shape

    for kernel_row in range(kernel_height):
        for kernel_column in range(kernel_width):
            image_row = start_row + kernel_row
            image_column = start_column + kernel_column

            pixel = image[image_row, image_column]
            weight = kernel[kernel_row, kernel_column]

            image_gradients[image_row, image_column] = output_gradient * weight
            kernel_gradients[kernel_row, kernel_column] = output_gradient * pixel

    bias_gradient = output_gradient

    return image_gradients, kernel_gradients, bias_gradient


def convolve_image_backward(image, kernel, output_gradients, stride=1, padding=0):
    padded_image = pad_image(image, padding)

    padded_gradients = np.zeros(padded_image.shape, dtype=float)
    kernel_gradients = np.zeros(kernel.shape, dtype=float)
    bias_gradient = 0.0

    output_height, output_width = output_gradients.shape
    kernel_height, kernel_width = kernel.shape

    for output_row in range(output_height):
        for output_column in range(output_width):
            start_row = output_row * stride
            start_column = output_column * stride

            image_contribution, kernel_contribution, bias_contribution = apply_kernel_backward(padded_image, kernel, start_row, start_column, output_gradients[output_row, output_column])

            for kernel_row in range(kernel_height):
                for kernel_column in range(kernel_width):
                    image_row = start_row + kernel_row
                    image_column = start_column + kernel_column

                    padded_gradients[image_row, image_column] += image_contribution[image_row, image_column]

                    kernel_gradients[kernel_row, kernel_column] += kernel_contribution[kernel_row, kernel_column]

            bias_gradient += bias_contribution

    height, width = image.shape
    image_gradients = np.empty(image.shape, dtype=float)

    for row in range(height):
        for column in range(width):
            image_gradients[row, column] = padded_gradients[row + padding, column + padding]

    return image_gradients, kernel_gradients, bias_gradient


def upsample_backward(output_gradients, scale=2):
    channels, output_height, output_width = output_gradients.shape
    blocks = output_gradients.reshape(
        channels, output_height // scale, scale, output_width // scale, scale
    )
    return np.sum(blocks, axis=(2, 4))



def dense_backward(inputs, weights, output_gradients):
    return (
        np.matmul(output_gradients, weights.T),
        np.outer(inputs, output_gradients),
        output_gradients.copy()
    )



def relu_backward(inputs, output_gradients):
    return np.where(inputs > 0, output_gradients, 0.0)



def activate_feature_maps_backward(inputs, output_gradients):
    return np.where(inputs > 0, output_gradients, 0.0)



def convolution_backward(inputs, kernels, output_gradients, stride=1, padding=0, cache=None):
    channels, height, width = inputs.shape
    filter_count, _, kernel_height, kernel_width = kernels.shape
    if cache is None:
        patches = image_to_patches(inputs, kernel_height, kernel_width, stride, padding)
    else:
        patches = cache["patches"]

    filter_weights = kernels_to_matrix(kernels)
    gradient_matrix = output_gradients.reshape(filter_count, -1).T
    weight_matrix_gradients = np.matmul(patches.T, gradient_matrix)
    patch_gradients = np.matmul(gradient_matrix, filter_weights.T)
    bias_gradients = np.sum(gradient_matrix, axis=0)
    kernel_gradients = weight_matrix_gradients.T.reshape(kernels.shape)

    indices, original_indices = _patch_indices(
        channels, height, width, kernel_height, kernel_width, stride, padding
    )
    padded_gradients = np.zeros(channels * (height + 2 * padding) * (width + 2 * padding))
    # add.at correctly sums repeated indices from overlapping patches.
    np.add.at(padded_gradients, indices.ravel(), patch_gradients.ravel())
    input_gradients = np.take(padded_gradients, original_indices)
    return input_gradients, kernel_gradients, bias_gradients



def sample_latent_backward(log_variance, noise, latent_gradients):
    return (
        latent_gradients.copy(),
        latent_gradients * noise * 0.5 * np.exp(0.5 * log_variance)
    )



def kl_loss_backward(mean, log_variance):
    return mean.copy(), 0.5 * (np.exp(log_variance) - 1.0)



def update_parameter(parameter, gradients, learning_rate):
    np.subtract(parameter, learning_rate * gradients, out=parameter)



def initialize_encoder(image_shape, filter_counts, latent_size, rng, kernel_size=3, stride=2, padding=1):
    channels, height, width = image_shape
    layers = []

    for filter_count in filter_counts:
        kernels, biases = initialize_convolution(channels, filter_count, kernel_size, rng)

        layers.append({"kernels": kernels, "biases": biases, "stride": stride, "padding": padding})

        height = convolution_output_size(height, kernel_size, stride, padding)
        width = convolution_output_size(width, kernel_size, stride, padding)
        channels = filter_count

    feature_count = channels * height * width

    mean_weights, mean_biases = initialize_dense(feature_count, latent_size, rng)

    log_variance_weights, log_variance_biases = initialize_dense(feature_count, latent_size, rng)

    return {"layers": layers, "feature_shape": (channels, height, width), "mean_weights": mean_weights, "mean_biases": mean_biases, "log_variance_weights": log_variance_weights, "log_variance_biases": log_variance_biases}


def encode(inputs, encoder):
    current = inputs
    layer_cache = []

    for layer in encoder["layers"]:
        convolution_cache = {}
        feature_maps = convolution(current, layer["kernels"], layer["biases"], stride=layer["stride"], padding=layer["padding"], cache=convolution_cache)

        layer_cache.append({"inputs": current, "feature_maps": feature_maps, "convolution": convolution_cache})

        current = activate_feature_maps(feature_maps)

    features = flatten(current)

    mean = dense(features, encoder["mean_weights"], encoder["mean_biases"])

    log_variance = dense(features, encoder["log_variance_weights"], encoder["log_variance_biases"])

    cache = {"layers": layer_cache, "features": features}

    return mean, log_variance, cache


def initialize_decoder(encoder, image_channels, latent_size, rng):
    channels, height, width = encoder["feature_shape"]
    feature_count = channels * height * width

    weights, biases = initialize_dense(latent_size, feature_count, rng)

    layers = []
    encoder_layers = encoder["layers"]

    for index in range(len(encoder_layers) - 1, -1, -1):
        encoder_layer = encoder_layers[index]

        if index == 0:
            filter_count = image_channels
        else:
            filter_count = encoder_layers[index - 1]["kernels"].shape[0]

        kernel_size = encoder_layer["kernels"].shape[2]

        kernels, convolution_biases = initialize_convolution(channels, filter_count, kernel_size, rng)

        layers.append({"kernels": kernels, "biases": convolution_biases, "scale": encoder_layer["stride"], "padding": kernel_size // 2})

        channels = filter_count

    return {"weights": weights, "biases": biases, "feature_shape": encoder["feature_shape"], "layers": layers}


def decode(latent, decoder):
    values = dense(latent, decoder["weights"], decoder["biases"])

    channels, height, width = decoder["feature_shape"]
    initial_maps = unflatten(values, channels, height, width)
    current = activate_feature_maps(initial_maps)

    layer_cache = []

    for index in range(len(decoder["layers"])):
        layer = decoder["layers"][index]

        enlarged_maps = upsample(current, scale=layer["scale"])

        convolution_cache = {}
        feature_maps = convolution(enlarged_maps, layer["kernels"], layer["biases"], stride=1, padding=layer["padding"], cache=convolution_cache)

        layer_cache.append({"inputs": enlarged_maps, "feature_maps": feature_maps, "convolution": convolution_cache})

        if index == len(decoder["layers"]) - 1:
            current = sigmoid(feature_maps)
        else:
            current = activate_feature_maps(feature_maps)

    cache = {"latent": latent, "initial_maps": initial_maps, "layers": layer_cache, "reconstruction": current}

    return current, cache


def decode_backward(reconstruction_gradients, decoder, cache):
    current_gradients = reconstruction_gradients
    layer_gradients = [None] * len(decoder["layers"])

    for index in range(len(decoder["layers"]) - 1, -1, -1):
        layer = decoder["layers"][index]
        saved = cache["layers"][index]

        if index == len(decoder["layers"]) - 1:
            feature_gradients = sigmoid_backward(cache["reconstruction"], current_gradients)
        else:
            feature_gradients = activate_feature_maps_backward(saved["feature_maps"], current_gradients)

        enlarged_gradients, kernel_gradients, bias_gradients = convolution_backward(saved["inputs"], layer["kernels"], feature_gradients, stride=1, padding=layer["padding"], cache=saved.get("convolution"))

        layer_gradients[index] = {"kernels": kernel_gradients, "biases": bias_gradients}

        current_gradients = upsample_backward(enlarged_gradients, scale=layer["scale"])

    initial_gradients = activate_feature_maps_backward(cache["initial_maps"], current_gradients)

    value_gradients = flatten(initial_gradients)

    latent_gradients, weight_gradients, bias_gradients = dense_backward(cache["latent"], decoder["weights"], value_gradients)

    parameter_gradients = {"weights": weight_gradients, "biases": bias_gradients, "layers": layer_gradients}

    return latent_gradients, parameter_gradients


def encode_backward(mean_gradients, log_variance_gradients, encoder, cache):
    mean_feature_gradients, mean_weight_gradients, mean_bias_gradients = dense_backward(cache["features"], encoder["mean_weights"], mean_gradients)

    variance_feature_gradients, variance_weight_gradients, variance_bias_gradients = dense_backward(cache["features"], encoder["log_variance_weights"], log_variance_gradients)

    feature_gradients = np.empty(cache["features"].shape, dtype=float)

    for index in range(len(feature_gradients)):
        feature_gradients[index] = mean_feature_gradients[index] + variance_feature_gradients[index]

    channels, height, width = encoder["feature_shape"]
    current_gradients = unflatten(feature_gradients, channels, height, width)

    layer_gradients = [None] * len(encoder["layers"])

    for index in range(len(encoder["layers"]) - 1, -1, -1):
        layer = encoder["layers"][index]
        saved = cache["layers"][index]

        convolution_gradients = activate_feature_maps_backward(saved["feature_maps"], current_gradients)

        current_gradients, kernel_gradients, bias_gradients = convolution_backward(saved["inputs"], layer["kernels"], convolution_gradients, stride=layer["stride"], padding=layer["padding"], cache=saved.get("convolution"))

        layer_gradients[index] = {"kernels": kernel_gradients, "biases": bias_gradients}

    parameter_gradients = {"mean_weights": mean_weight_gradients, "mean_biases": mean_bias_gradients, "log_variance_weights": variance_weight_gradients, "log_variance_biases": variance_bias_gradients, "layers": layer_gradients}

    return current_gradients, parameter_gradients


def update_encoder(encoder, gradients, learning_rate):
    dense_parameters = ("mean_weights", "mean_biases", "log_variance_weights", "log_variance_biases")

    for name in dense_parameters:
        update_parameter(encoder[name], gradients[name], learning_rate)

    for index in range(len(encoder["layers"])):
        layer = encoder["layers"][index]
        layer_gradients = gradients["layers"][index]

        update_parameter(layer["kernels"], layer_gradients["kernels"], learning_rate)
        update_parameter(layer["biases"], layer_gradients["biases"], learning_rate)


def update_decoder(decoder, gradients, learning_rate):
    update_parameter(decoder["weights"], gradients["weights"], learning_rate)

    update_parameter(decoder["biases"], gradients["biases"], learning_rate)

    for index in range(len(decoder["layers"])):
        layer = decoder["layers"][index]
        layer_gradients = gradients["layers"][index]

        update_parameter(layer["kernels"], layer_gradients["kernels"], learning_rate)
        update_parameter(layer["biases"], layer_gradients["biases"], learning_rate)


def train_step(inputs, encoder, decoder, rng, learning_rate, beta):
    # Forward pass.
    mean, log_variance, encoder_cache = encode(inputs, encoder)
    latent, noise = sample_latent(mean, log_variance, rng)
    reconstruction, decoder_cache = decode(latent, decoder)

    # Calculate losses.
    pixel_loss = reconstruction_loss(inputs, reconstruction)
    latent_loss = kl_loss(mean, log_variance)
    total_loss = pixel_loss + beta * latent_loss

    # Backward through the decoder.
    reconstruction_gradients = reconstruction_loss_backward(inputs, reconstruction)

    latent_gradients, decoder_gradients = decode_backward(reconstruction_gradients, decoder, decoder_cache)

    # Backward through sampling and KL loss.
    mean_gradients, log_variance_gradients = sample_latent_backward(log_variance, noise, latent_gradients)

    kl_mean_gradients, kl_log_variance_gradients = kl_loss_backward(mean, log_variance)

    for index in range(len(mean)):
        mean_gradients[index] += beta * kl_mean_gradients[index]
        log_variance_gradients[index] += beta * kl_log_variance_gradients[index]

    # Backward through the encoder.
    _, encoder_gradients = encode_backward(mean_gradients, log_variance_gradients, encoder, encoder_cache)

    # Update parameters after all gradients are calculated.
    update_encoder(encoder, encoder_gradients, learning_rate)
    update_decoder(decoder, decoder_gradients, learning_rate)

    return total_loss, pixel_loss, latent_loss


@lru_cache(maxsize=64)
def _patch_indices(channels, height, width, kernel_height, kernel_width, stride, padding):
    """Cache coordinates, not image values. Loops run once per layer shape."""
    output_height = convolution_output_size(height, kernel_height, stride, padding)
    output_width = convolution_output_size(width, kernel_width, stride, padding)
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    indices = np.empty((output_height * output_width, channels * kernel_height * kernel_width), dtype=np.intp)
    position = 0
    for output_row in range(output_height):
        for output_column in range(output_width):
            feature = 0
            for channel in range(channels):
                for kernel_row in range(kernel_height):
                    for kernel_column in range(kernel_width):
                        row = output_row * stride + kernel_row
                        column = output_column * stride + kernel_column
                        indices[position, feature] = (channel * padded_height + row) * padded_width + column
                        feature += 1
            position += 1

    # Coordinates selecting the original image from its padded gradient.
    original_indices = np.empty((channels, height, width), dtype=np.intp)
    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                original_indices[channel, row, column] = (
                    (channel * padded_height + row + padding) * padded_width
                    + column + padding
                )
    return indices, original_indices



def image_to_patches(inputs, kernel_height, kernel_width, stride, padding):
    channels, height, width = inputs.shape
    indices, _ = _patch_indices(channels, height, width, kernel_height, kernel_width, stride, padding)
    padded = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding)))
    return np.take(padded, indices)



def kernels_to_matrix(kernels):
    filter_count = kernels.shape[0]
    return kernels.reshape(filter_count, -1).T

