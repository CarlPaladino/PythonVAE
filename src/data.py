import csv

import numpy as np


def load_images(path, height=28, width=28, limit=None):
    images = []
    labels = []

    with open(path, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if limit is not None and len(images) >= limit:
                break

            label = int(row[0])
            image = np.empty((1, height, width), dtype=float)

            for image_row in range(height):
                for image_column in range(width):
                    pixel_index = 1 + image_column * height + image_row
                    pixel = float(row[pixel_index])

                    image[0, image_row, image_column] = pixel / 255.0

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels, dtype=int)
