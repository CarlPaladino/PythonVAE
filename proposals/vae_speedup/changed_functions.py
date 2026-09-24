import numpy as np

# Review excerpt; full runnable module is vae.py in this folder.

def image_to_patches(inputs, kernel_height, kernel_width, stride, padding):
    """One row per output position, one column per channel/kernel coordinate."""
    channels, height, width = inputs.shape
    output_height = convolution_output_size(height, kernel_height, stride, padding)
    output_width = convolution_output_size(width, kernel_width, stride, padding)
    patch_size = channels * kernel_height * kernel_width
    patches = np.empty((output_height * output_width, patch_size))

    for channel in range(channels):
        padded = pad_image(inputs[channel], padding)
        position = 0
        for output_row in range(output_height):
            for output_column in range(output_width):
                feature = channel * kernel_height * kernel_width
                for kernel_row in range(kernel_height):
                    for kernel_column in range(kernel_width):
                        row = output_row * stride + kernel_row
                        column = output_column * stride + kernel_column
                        patches[position, feature] = padded[row, column]
                        feature += 1
                position += 1

    return patches


def kernels_to_matrix(kernels):
    """Each column contains one filter, in the same order as a patch."""
    filters, channels, kernel_height, kernel_width = kernels.shape
    matrix = np.empty((channels * kernel_height * kernel_width, filters))

    for filter_index in range(filters):
        feature = 0
        for channel in range(channels):
            for row in range(kernel_height):
                for column in range(kernel_width):
                    matrix[feature, filter_index] = kernels[filter_index, channel, row, column]
                    feature += 1

    return matrix


def convolution(inputs, kernels, biases, stride=1, padding=0):
    channels, height, width = inputs.shape
    filter_count, _, kernel_height, kernel_width = kernels.shape
    output_height = convolution_output_size(height, kernel_height, stride, padding)
    output_width = convolution_output_size(width, kernel_width, stride, padding)

    patches = image_to_patches(inputs, kernel_height, kernel_width, stride, padding)
    filter_weights = kernels_to_matrix(kernels)
    weighted_sums = np.matmul(patches, filter_weights)

    output = np.empty((filter_count, output_height, output_width))
    position = 0
    for row in range(output_height):
        for column in range(output_width):
            for filter_index in range(filter_count):
                output[filter_index, row, column] = weighted_sums[position, filter_index] + biases[filter_index]
            position += 1

    return output


def convolution_backward(inputs, kernels, output_gradients, stride=1, padding=0):
    channels, height, width = inputs.shape
    filter_count, _, kernel_height, kernel_width = kernels.shape
    _, output_height, output_width = output_gradients.shape

    patches = image_to_patches(inputs, kernel_height, kernel_width, stride, padding)
    filter_weights = kernels_to_matrix(kernels)
    gradient_matrix = np.empty((output_height * output_width, filter_count))
    bias_gradients = np.zeros(filter_count)

    position = 0
    for row in range(output_height):
        for column in range(output_width):
            for filter_index in range(filter_count):
                gradient = output_gradients[filter_index, row, column]
                gradient_matrix[position, filter_index] = gradient
                bias_gradients[filter_index] += gradient
            position += 1

    # Add contributions from all spatial positions for each weight.
    weight_matrix_gradients = np.matmul(patches.T, gradient_matrix)
    # Add contributions from all filters for each value in a patch.
    patch_gradients = np.matmul(gradient_matrix, filter_weights.T)

    kernel_gradients = np.empty(kernels.shape)
    for filter_index in range(filter_count):
        feature = 0
        for channel in range(channels):
            for row in range(kernel_height):
                for column in range(kernel_width):
                    kernel_gradients[filter_index, channel, row, column] = weight_matrix_gradients[feature, filter_index]
                    feature += 1

    padded_gradients = np.zeros((channels, height + 2 * padding, width + 2 * padding))
    position = 0
    for output_row in range(output_height):
        for output_column in range(output_width):
            feature = 0
            for channel in range(channels):
                for kernel_row in range(kernel_height):
                    for kernel_column in range(kernel_width):
                        row = output_row * stride + kernel_row
                        column = output_column * stride + kernel_column
                        # Overlapping patches must add into the same input pixel.
                        padded_gradients[channel, row, column] += patch_gradients[position, feature]
                        feature += 1
            position += 1

    input_gradients = np.empty(inputs.shape)
    for channel in range(channels):
        for row in range(height):
            for column in range(width):
                input_gradients[channel, row, column] = padded_gradients[channel, row + padding, column + padding]

    return input_gradients, kernel_gradients, bias_gradients


def dense_backward(inputs, weights, output_gradients):
    input_gradients = np.matmul(output_gradients, weights.T)
    weight_gradients = np.outer(inputs, output_gradients)
    bias_gradients = output_gradients.copy()
    return input_gradients, weight_gradients, bias_gradients


def update_parameter(parameter, gradients, learning_rate):
    # Elementwise SGD; update the existing array rather than replacing it.
    np.subtract(parameter, learning_rate * gradients, out=parameter)
