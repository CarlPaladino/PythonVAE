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
    height, width = image.shape

    padded_height = height + 2 * padding
    padded_width = width + 2 * padding

    padded_image = np.zeros((padded_height, padded_width), dtype=image.dtype)

    for row in range(height):
        for column in range(width):
            padded_image[row + padding, column + padding] = image[row, column]

    return padded_image


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
    height, width = feature_map.shape
    output = np.zeros((height, width), dtype=feature_map.dtype)

    for row in range(height):
        for column in range(width):
            value = feature_map[row, column]

            if value > 0:
                output[row, column] = value

    return output


def flatten(feature_maps):
    channels, height, width = feature_maps.shape
    output = np.empty(channels * height * width, dtype=feature_maps.dtype)

    index = 0

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                output[index] = feature_maps[channel, row, column]
                index += 1

    return output


def dense(inputs, weights, biases):
    output = np.matmul(inputs, weights)

    for neuron in range(len(biases)):
        output[neuron] += biases[neuron]

    return output


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


def convolution(inputs, kernels, biases, stride=1, padding=0):
    input_channels, height, width = inputs.shape
    filter_count, _, kernel_height, kernel_width = kernels.shape

    output_height = convolution_output_size(height, kernel_height, stride, padding)
    output_width = convolution_output_size(width, kernel_width, stride, padding)

    output = np.zeros((filter_count, output_height, output_width))

    for filter_index in range(filter_count):
        for channel in range(input_channels):
            channel_output = convolve_image(inputs[channel], kernels[filter_index, channel], bias=0.0, stride=stride, padding=padding)

            for row in range(output_height):
                for column in range(output_width):
                    output[filter_index, row, column] += channel_output[row, column]

        for row in range(output_height):
            for column in range(output_width):
                output[filter_index, row, column] += biases[filter_index]

    return output


def activate_feature_maps(feature_maps):
    channels, height, width = feature_maps.shape
    output = np.empty(feature_maps.shape, dtype=feature_maps.dtype)

    for channel in range(channels):
        activated_map = relu(feature_maps[channel])

        for row in range(height):
            for column in range(width):
                output[channel, row, column] = activated_map[row, column]

    return output


def sample_latent(mean, log_variance, rng):
    latent_size = len(mean)

    noise = rng.normal(0.0, 1.0, size=latent_size)
    latent = np.empty(latent_size)

    for index in range(latent_size):
        standard_deviation = np.exp(0.5 * log_variance[index])

        latent[index] = mean[index] + standard_deviation * noise[index]

    return latent, noise


def unflatten(values, channels, height, width):
    output = np.empty((channels, height, width), dtype=values.dtype)

    index = 0

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                output[channel, row, column] = values[index]
                index += 1

    return output


def upsample(feature_maps, scale=2):
    channels, height, width = feature_maps.shape

    output_height = height * scale
    output_width = width * scale

    output = np.empty((channels, output_height, output_width), dtype=feature_maps.dtype)

    for channel in range(channels):
        for row in range(output_height):
            for column in range(output_width):
                input_row = row // scale
                input_column = column // scale

                output[channel, row, column] = feature_maps[channel, input_row, input_column]

    return output


def sigmoid(feature_maps):
    channels, height, width = feature_maps.shape
    output = np.empty(feature_maps.shape, dtype=float)

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                value = feature_maps[channel, row, column]

                if value >= 0:
                    output[channel, row, column] = 1.0 / (1.0 + np.exp(-value))
                else:
                    exponential = np.exp(value)
                    output[channel, row, column] = exponential / (1.0 + exponential)

    return output


def reconstruction_loss(original, reconstruction):
    channels, height, width = original.shape
    total = 0.0

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                difference = reconstruction[channel, row, column] - original[channel, row, column]

                total += 0.5 * difference * difference

    return total


def kl_loss(mean, log_variance):
    total = 0.0

    for index in range(len(mean)):
        variance = np.exp(log_variance[index])

        total += 0.5 * (mean[index] * mean[index] + variance - 1.0 - log_variance[index])

    return total


def reconstruction_loss_backward(original, reconstruction):
    channels, height, width = original.shape
    gradients = np.empty(original.shape, dtype=float)

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                gradients[channel, row, column] = reconstruction[channel, row, column] - original[channel, row, column]

    return gradients


def sigmoid_backward(sigmoid_output, output_gradients):
    channels, height, width = sigmoid_output.shape
    input_gradients = np.empty(sigmoid_output.shape, dtype=float)

    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                value = sigmoid_output[channel, row, column]
                slope = value * (1.0 - value)

                input_gradients[channel, row, column] = output_gradients[channel, row, column] * slope

    return input_gradients


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

    input_height = output_height // scale
    input_width = output_width // scale

    input_gradients = np.zeros((channels, input_height, input_width), dtype=float)

    for channel in range(channels):
        for row in range(output_height):
            for column in range(output_width):
                input_row = row // scale
                input_column = column // scale

                input_gradients[channel, input_row, input_column] += output_gradients[channel, row, column]

    return input_gradients


def dense_backward(inputs, weights, output_gradients):
    input_count, neuron_count = weights.shape

    input_gradients = np.zeros(input_count)
    weight_gradients = np.empty(weights.shape, dtype=float)
    bias_gradients = np.empty(neuron_count)

    for neuron in range(neuron_count):
        gradient = output_gradients[neuron]
        bias_gradients[neuron] = gradient

        for input_index in range(input_count):
            weight_gradients[input_index, neuron] = inputs[input_index] * gradient

            input_gradients[input_index] += weights[input_index, neuron] * gradient

    return input_gradients, weight_gradients, bias_gradients


def relu_backward(inputs, output_gradients):
    height, width = inputs.shape
    input_gradients = np.empty(inputs.shape, dtype=float)

    for row in range(height):
        for column in range(width):
            if inputs[row, column] > 0:
                input_gradients[row, column] = output_gradients[row, column]
            else:
                input_gradients[row, column] = 0.0

    return input_gradients


def activate_feature_maps_backward(inputs, output_gradients):
    channels, height, width = inputs.shape
    input_gradients = np.empty(inputs.shape, dtype=float)

    for channel in range(channels):
        channel_gradients = relu_backward(inputs[channel], output_gradients[channel])

        for row in range(height):
            for column in range(width):
                input_gradients[channel, row, column] = channel_gradients[row, column]

    return input_gradients


def convolution_backward(inputs, kernels, output_gradients, stride=1, padding=0):
    input_channels, height, width = inputs.shape
    filter_count, _, kernel_height, kernel_width = kernels.shape
    _, output_height, output_width = output_gradients.shape

    input_gradients = np.zeros(inputs.shape, dtype=float)
    kernel_gradients = np.empty(kernels.shape, dtype=float)
    bias_gradients = np.zeros(filter_count)

    for filter_index in range(filter_count):
        for channel in range(input_channels):
            image_gradient, kernel_gradient, _ = convolve_image_backward(inputs[channel], kernels[filter_index, channel], output_gradients[filter_index], stride=stride, padding=padding)

            for row in range(height):
                for column in range(width):
                    input_gradients[channel, row, column] += image_gradient[row, column]

            for row in range(kernel_height):
                for column in range(kernel_width):
                    kernel_gradients[filter_index, channel, row, column] = kernel_gradient[row, column]

        for row in range(output_height):
            for column in range(output_width):
                bias_gradients[filter_index] += output_gradients[filter_index, row, column]

    return input_gradients, kernel_gradients, bias_gradients


def sample_latent_backward(log_variance, noise, latent_gradients):
    latent_size = len(log_variance)

    mean_gradients = np.empty(latent_size)
    log_variance_gradients = np.empty(latent_size)

    for index in range(latent_size):
        standard_deviation = np.exp(0.5 * log_variance[index])

        mean_gradients[index] = latent_gradients[index]

        log_variance_gradients[index] = latent_gradients[index] * noise[index] * 0.5 * standard_deviation

    return mean_gradients, log_variance_gradients


def kl_loss_backward(mean, log_variance):
    latent_size = len(mean)

    mean_gradients = np.empty(latent_size)
    log_variance_gradients = np.empty(latent_size)

    for index in range(latent_size):
        mean_gradients[index] = mean[index]

        log_variance_gradients[index] = 0.5 * (np.exp(log_variance[index]) - 1.0)

    return mean_gradients, log_variance_gradients


def update_parameter(parameter, gradients, learning_rate):
    for index in np.ndindex(parameter.shape):  # Visit every index in parameter's shape
        parameter[index] -= learning_rate * gradients[index]


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
        feature_maps = convolution(current, layer["kernels"], layer["biases"], stride=layer["stride"], padding=layer["padding"])

        layer_cache.append({"inputs": current, "feature_maps": feature_maps})

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

        feature_maps = convolution(enlarged_maps, layer["kernels"], layer["biases"], stride=1, padding=layer["padding"])

        layer_cache.append({"inputs": enlarged_maps, "feature_maps": feature_maps})

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

        enlarged_gradients, kernel_gradients, bias_gradients = convolution_backward(saved["inputs"], layer["kernels"], feature_gradients, stride=1, padding=layer["padding"])

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

        current_gradients, kernel_gradients, bias_gradients = convolution_backward(saved["inputs"], layer["kernels"], convolution_gradients, stride=layer["stride"], padding=layer["padding"])

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
