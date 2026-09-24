from pathlib import Path

from src.vae import decode, encode


def save_reconstructions(path, inputs, encoder, decoder, epoch, sample_count=5):
    sample_count = min(sample_count, len(inputs))
    _, height, width = inputs[0].shape
    scale = 5
    left_margin = 190
    cell_width = width * scale + 20
    cell_height = height * scale + 40
    canvas_width = left_margin + sample_count * cell_width
    canvas_height = 70 + 2 * cell_height

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}">\n')
        file.write('<rect width="100%" height="100%" fill="white"/>\n')
        file.write(f'<text x="20" y="25" font-family="sans-serif" font-size="18">Epoch {epoch}: training-image reconstructions</text>\n')
        file.write('<text x="20" y="95" font-family="sans-serif" font-size="16">Original</text>\n')
        file.write(f'<text x="20" y="{95 + cell_height}" font-family="sans-serif" font-size="16">Decoded mean</text>\n')

        for sample_index in range(sample_count):
            original = inputs[sample_index]
            mean, _, _ = encode(original, encoder)
            reconstruction, _ = decode(mean, decoder)
            left = left_margin + sample_index * cell_width
            file.write(f'<text x="{left}" y="55" font-family="sans-serif" font-size="14">Sample {sample_index + 1}</text>\n')

            images = (original, reconstruction)
            for image_index in range(2):
                image = images[image_index]
                top = 70 + image_index * cell_height

                for row in range(height):
                    for column in range(width):
                        shade = round(float(image[0, row, column]) * 255)
                        x = left + column * scale
                        y = top + row * scale
                        file.write(f'<rect x="{x}" y="{y}" width="{scale}" height="{scale}" fill="rgb({shade},{shade},{shade})"/>\n')

        file.write('</svg>\n')
