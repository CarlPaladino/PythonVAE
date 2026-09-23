import numpy as np


class Dense:
    def __init__(self, input_dim, output_dim, rng, dtype=np.float32):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Layer dimensions must be positive.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inputs = None
        self.dweights = None
        self.dbiases = None

        weight_scale = np.sqrt(2.0 / input_dim)
        self.weights = rng.normal(0.0, weight_scale, (input_dim, output_dim)).astype(dtype)
        self.biases = np.zeros((1, output_dim), dtype=dtype)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("Expected inputs shaped (samples, input_dim).")
        self.inputs = inputs

        weighted_inputs = np.matmul(inputs, self.weights)
        outputs = weighted_inputs + self.biases
        return outputs

    def backward(self, output_gradients):
        if self.inputs is None:
            raise RuntimeError("Call forward before backward.")
        expected_shape = (self.inputs.shape[0], self.output_dim)
        if output_gradients.shape != expected_shape:
            raise ValueError("Dense gradient shape mismatch.")

        inputs_by_feature = self.inputs.T
        self.dweights = np.matmul(inputs_by_feature, output_gradients)
        self.dbiases = np.sum(output_gradients, axis=0, keepdims=True)

        weights_by_neuron = self.weights.T
        input_gradients = np.matmul(output_gradients, weights_by_neuron)
        return input_gradients


class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size, rng, stride=1, padding=0, dtype=np.float32):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = np.dtype(dtype)

        fan_in = input_channels * kernel_size * kernel_size
        weight_shape = (output_channels, input_channels, kernel_size, kernel_size)
        self.weights = rng.normal(0.0, np.sqrt(2.0 / fan_in), weight_shape).astype(self.dtype)
        self.biases = np.zeros(output_channels, dtype=self.dtype)
        self.dweights = None
        self.dbiases = None
        self.patch_matrix = None
        self.window_pixel_indices = None
        self.padded_shape = None
        self.cached_image_shape = None
        self.input_shape = None
        self.output_shape = None

    def prepare_window_indices(self, height, width):
        image_shape = (height, width, self.kernel_size, self.stride, self.padding)
        if self.cached_image_shape == image_shape:
            return

        padded_width = width + 2 * self.padding
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        window_count = output_height * output_width
        kernel_pixel_count = self.kernel_size * self.kernel_size

        window_rows = np.repeat(np.arange(output_height), output_width) * self.stride
        window_columns = np.tile(np.arange(output_width), output_height) * self.stride
        window_starts = window_rows * padded_width + window_columns

        self.window_pixel_indices = np.empty((kernel_pixel_count, window_count), dtype=np.intp)
        for kernel_row in range(self.kernel_size):
            for kernel_column in range(self.kernel_size):
                kernel_position = kernel_row * self.kernel_size + kernel_column
                pixel_offset = kernel_row * padded_width + kernel_column
                self.window_pixel_indices[kernel_position] = window_starts + pixel_offset
        self.cached_image_shape = image_shape

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=self.dtype)
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError("Expected inputs shaped (samples, input_channels, height, width).")
        batch_size, _, height, width = inputs.shape
        if min(height, width) <= 0 or min(height, width) + 2 * self.padding < self.kernel_size:
            raise ValueError("Kernel must fit within a nonempty padded image.")

        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        window_count = output_height * output_width
        kernel_pixel_count = self.kernel_size * self.kernel_size
        filter_input_count = self.input_channels * kernel_pixel_count
        self.input_shape = inputs.shape
        self.output_shape = (batch_size, self.output_channels, output_height, output_width)
        self.prepare_window_indices(height, width)

        padding_per_axis = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        padded_inputs = np.pad(inputs, padding_per_axis)
        self.padded_shape = padded_inputs.shape
        padded_pixel_count = self.padded_shape[2] * self.padded_shape[3]
        flat_images = padded_inputs.reshape(batch_size, self.input_channels, padded_pixel_count)

        indices_by_window = self.window_pixel_indices.T
        patches = np.take(flat_images, indices_by_window, axis=2)

        patches_by_window = np.transpose(patches, (0, 2, 1, 3))
        self.patch_matrix = patches_by_window.reshape(batch_size * window_count, filter_input_count)
        filter_matrix = self.weights.reshape(self.output_channels, filter_input_count)

        output_matrix = np.matmul(self.patch_matrix, filter_matrix.T)
        output_matrix += self.biases
        output_images = output_matrix.reshape(batch_size, output_height, output_width, self.output_channels)
        return np.transpose(output_images, (0, 3, 1, 2))

    def backward(self, output_gradients):
        if self.patch_matrix is None:
            raise RuntimeError("Call forward before backward.")
        output_gradients = np.asarray(output_gradients, dtype=self.dtype)
        if output_gradients.shape != self.output_shape:
            raise ValueError("Conv2D gradient shape mismatch.")

        batch_size, _, output_height, output_width = self.output_shape
        window_count = output_height * output_width
        kernel_pixel_count = self.kernel_size * self.kernel_size
        filter_input_count = self.input_channels * kernel_pixel_count

        gradients_by_window = np.transpose(output_gradients, (0, 2, 3, 1))
        gradient_matrix = gradients_by_window.reshape(batch_size * window_count, self.output_channels)
        weight_gradients = np.matmul(gradient_matrix.T, self.patch_matrix)
        self.dweights = weight_gradients.reshape(self.weights.shape)
        self.dbiases = np.sum(gradient_matrix, axis=0)

        filter_matrix = self.weights.reshape(self.output_channels, filter_input_count)
        patch_gradient_matrix = np.matmul(gradient_matrix, filter_matrix)
        patch_gradients = patch_gradient_matrix.reshape(batch_size, window_count, self.input_channels, kernel_pixel_count)
        gradients_by_channel = np.transpose(patch_gradients, (0, 2, 3, 1))

        group_count = batch_size * self.input_channels
        gradients_by_kernel = np.ascontiguousarray(gradients_by_channel).reshape(group_count, kernel_pixel_count, window_count)
        padded_pixel_count = self.padded_shape[2] * self.padded_shape[3]
        flat_input_gradients = np.zeros(group_count * padded_pixel_count, dtype=self.dtype)
        group_starts = (np.arange(group_count) * padded_pixel_count).reshape(group_count, 1)

        for kernel_position in range(kernel_pixel_count):
            window_addresses = self.window_pixel_indices[kernel_position].reshape(1, window_count)
            destination_addresses = group_starts + window_addresses
            contributions = np.take(gradients_by_kernel, kernel_position, axis=1)
            flat_input_gradients[destination_addresses] += contributions

        padded_gradients = flat_input_gradients.reshape(self.padded_shape)
        _, _, height, width = self.input_shape
        image_rows = np.arange(height) + self.padding
        image_columns = np.arange(width) + self.padding
        gradients_without_border_rows = np.take(padded_gradients, image_rows, axis=2)
        return np.take(gradients_without_border_rows, image_columns, axis=3)


class Upsample2D:
    def __init__(self):
        self.input_shape = None

    def forward(self, inputs):
        if inputs.ndim != 4:
            raise ValueError("Expected inputs shaped (samples, channels, height, width).")
        self.input_shape = inputs.shape
        doubled_rows = np.repeat(inputs, 2, axis=2)
        return np.repeat(doubled_rows, 2, axis=3)

    def backward(self, output_gradients):
        if self.input_shape is None:
            raise RuntimeError("Call forward before backward.")
        batch_size, channels, height, width = self.input_shape
        if output_gradients.shape != (batch_size, channels, height * 2, width * 2):
            raise ValueError("Upsample2D gradient shape mismatch.")

        pixel_copies = output_gradients.reshape(batch_size, channels, height, 2, width, 2)
        return np.sum(pixel_copies, axis=(3, 5))


class ReLU:
    def __init__(self):
        self.positive_inputs = None

    def forward(self, inputs):
        self.positive_inputs = inputs > 0.0
        outputs = np.maximum(inputs, 0.0)
        return outputs

    def backward(self, output_gradients):
        if self.positive_inputs is None:
            raise RuntimeError("Call forward before backward.")
        if output_gradients.shape != self.positive_inputs.shape:
            raise ValueError("ReLU gradient shape mismatch.")
        input_gradients = output_gradients * self.positive_inputs
        return input_gradients


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        negative_magnitude = -np.abs(inputs)
        exponential = np.exp(negative_magnitude)
        denominator = 1.0 + exponential
        positive_result = 1.0 / denominator
        negative_result = exponential / denominator
        self.output = np.where(inputs >= 0.0, positive_result, negative_result)
        return self.output

    def backward(self, output_gradients):
        if self.output is None:
            raise RuntimeError("Call forward before backward.")
        if output_gradients.shape != self.output.shape:
            raise ValueError("Sigmoid gradient shape mismatch.")
        sigmoid_derivative = self.output * (1.0 - self.output)
        input_gradients = output_gradients * sigmoid_derivative
        return input_gradients


class Reparameterization:
    def __init__(self, rng):
        self.rng = rng
        self.epsilon = None
        self.standard_deviation = None

    def forward(self, mean, log_variance):
        if mean.shape != log_variance.shape:
            raise ValueError("Mean and log variance shapes must match.")
        log_standard_deviation = 0.5 * log_variance
        self.standard_deviation = np.exp(log_standard_deviation)
        self.epsilon = self.rng.standard_normal(mean.shape, dtype=mean.dtype)
        scaled_noise = self.standard_deviation * self.epsilon
        latent_values = mean + scaled_noise
        return latent_values

    def backward(self, latent_gradients):
        if self.epsilon is None:
            raise RuntimeError("Call forward before backward.")
        if latent_gradients.shape != self.epsilon.shape:
            raise ValueError("Latent gradient shape mismatch.")
        mean_gradients = latent_gradients.copy()
        deviation_gradients = latent_gradients * self.epsilon
        log_variance_gradients = deviation_gradients * 0.5 * self.standard_deviation
        return mean_gradients, log_variance_gradients
