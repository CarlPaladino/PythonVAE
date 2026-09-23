import numpy as np


def decode_rows(rows, height, width):
    pixel_count = height * width
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != pixel_count + 1:
        raise ValueError("Expected nonempty CSV rows containing a label and height * width pixels.")
    labels = np.take(rows, 0, axis=1).astype(np.int64)
    pixel_columns = np.arange(1, pixel_count + 1)
    pixels = np.take(rows, pixel_columns, axis=1)
    if np.any(pixels < 0) or np.any(pixels > 255):
        raise ValueError("Pixels must be between 0 and 255.")

    sample_count = rows.shape[0]
    columns_first = pixels.reshape(sample_count, width, height)
    upright_images = np.transpose(columns_first, axes=(0, 2, 1))
    images = np.ascontiguousarray(upright_images, dtype=np.float32)
    images /= 255.0
    return images, labels


def decode_row(row, height, width):
    rows = np.asarray([row], dtype=np.float32)
    images, labels = decode_rows(rows, height, width)
    return images[0], int(labels[0])


def load_images(path, height, width, limit=None):
    if limit is not None and limit <= 0:
        raise ValueError("Sample limit must be positive.")
    rows = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2, max_rows=limit)
    return decode_rows(rows, height, width)


def load_sample(path, sample_index, height, width):
    if sample_index < 0:
        raise ValueError("Sample index cannot be negative.")
    rows = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2, skiprows=sample_index, max_rows=1)
    if rows.size == 0:
        raise IndexError("Sample index is outside the dataset.")
    images, labels = decode_rows(rows, height, width)
    return images[0], int(labels[0])


def flatten_images(images):
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("Expected images shaped (samples, height, width).")
    sample_count, height, width = images.shape
    pixel_count = height * width
    return images.reshape(sample_count, pixel_count)


def unflatten_image(pixels, height, width):
    pixels = np.asarray(pixels)
    if pixels.ndim != 1 or pixels.size != height * width:
        raise ValueError("Pixel count does not match image dimensions.")
    return pixels.reshape(height, width)
