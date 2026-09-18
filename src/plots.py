from html import escape
from pathlib import Path


def text(x, y, label):
    return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="14">{escape(str(label))}</text>'


def image_elements(image, left, top, scale=5):
    elements = []
    for y in range(len(image)):
        for x in range(len(image[y])):
            shade = round(max(0.0, min(1.0, image[y][x])) * 255)
            elements.append(
                f'<rect x="{left + x * scale}" y="{top + y * scale}" '
                f'width="{scale}" height="{scale}" fill="rgb({shade},{shade},{shade})"/>'
            )
    return elements


def save_svg(path, width, height, elements):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        destination.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        destination.write('<rect width="100%" height="100%" fill="white"/>\n')
        for element in elements:
            destination.write(element)
            destination.write("\n")
        destination.write("</svg>\n")


def image_grid(images, rows, columns, path, title, labels=None):
    if rows <= 0 or columns <= 0 or len(images) != rows * columns:
        raise ValueError("Image count must equal positive rows * columns.")
    cell_width = len(images[0][0]) * 5 + 40
    cell_height = len(images[0]) * 5 + 40
    elements = [text(20, 25, title)]
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            left = 20 + column * cell_width
            top = 60 + row * cell_height
            if labels is not None:
                elements.append(text(left, top - 8, labels[index]))
            elements.extend(image_elements(images[index], left, top))
    save_svg(path, columns * cell_width + 20, rows * cell_height + 60, elements)


def bar_elements(values, left, top, title):
    elements = [text(left, top, title)]
    maximum = 1.0
    for value in values:
        maximum = max(maximum, abs(value))
    baseline = top + 110
    bar_width = 300 / len(values)
    elements.append(f'<path d="M {left} {baseline} h 300" stroke="black"/>')
    elements.append(text(left, top + 20, f"Symmetric scale: +/- {maximum:.3g}"))
    for index in range(len(values)):
        value = values[index]
        height = abs(value) / maximum * 65
        y = baseline
        if value >= 0:
            y = baseline - height
        x = left + index * bar_width
        elements.append(
            f'<rect x="{x}" y="{y}" width="{bar_width * 0.8}" height="{height}" fill="steelblue">'
            f'<title>Component {index + 1}: {value:.6g}</title></rect>'
        )
        elements.append(text(x, top + 195, index + 1))
    return elements
