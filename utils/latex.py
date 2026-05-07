from PIL import Image
import re
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from rectpack import newPacker
import rectpack


# Default colors used to simulate line artifacts/noise in the background
DEFAULT_LINE_COLORS = [(240, 0, 0, 150), (0, 0, 0, 180), (30, 30, 30, 150)]


def crop_to_formula(
    image, padding=30, max_width=6, line_colors=DEFAULT_LINE_COLORS, max_lines=5
):
    """
    Crops the image to tightly fit the mathematical formula and randomly adds
    background line artifacts to augment the data for better model robustness.
    """
    # Image is expected to be a 4-channel image with alpha.
    # Convert black pixels to white pixels.
    data = np.array(image)
    red, green, blue, alpha = data.T
    black_areas = (red < 10) & (blue < 10) & (green < 10)

    # Convert alpha to white.
    data[..., -1] = 255

    # Crop a box around the area that contains black pixels.
    coords = np.argwhere(black_areas)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1

    # Add padding.
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)

    # --- Image Augmentation ---
    # Randomly draw lines across the image to simulate ruled paper or stray marks
    image_cropped = data[y0:y1, x0:x1]
    mask = np.zeros_like(image_cropped)

    rotate_rads = np.random.random() * 10 - 5
    line_width = np.random.randint(1, max_width)
    num_lines = np.random.randint(1, max_lines)

    for _ in range(num_lines):
        center_y = np.random.randint(0, mask.shape[0])
        y_start = max(0, center_y - line_width // 2)
        y_end = min(mask.shape[0], center_y + line_width // 2)
        mask[y_start:y_end, :] = 1

    line_color = random.choice(line_colors)
    mask = rotate(mask, rotate_rads, reshape=False)

    # Apply the line color where the mask is active, preserving original image elsewhere
    augmented_image = image_cropped * (mask == 0) + mask * line_color

    result_image = Image.fromarray(augmented_image.astype("uint8"), "RGBA")
    return result_image.convert("RGB")


def renderedLaTeXLabelstr2Formula(label: str):
    # We're matching \\label{...whatever} and removing it
    label = re.sub(r"\\label\{[^\}]*\}", "", label)
    # We match \, and remove it.
    label = re.sub(r"\\,", "", label)
    return label


def display_formula(latex: str):
    # Remove \mbox{...} which is a formatting artifact
    parsed_latex = re.sub(r"\\mbox\{[^\}]*\}", "", latex)
    print(f"Formula: {parsed_latex}")


def generate_page(images, p_lines=.5, in_scale=100, paper_width=8.5, paper_height=11):
    """
    takes in a list of PIL images and returns an image with them pasted on
    and pixel wise bounding boxes describing the top left and bottom right
    coordinates of the equation on the paper of the form (x1,y1,x2,y2)
    """
    color_channel = 'RGBA'

    bounding_boxes =  [None] * len(images)
    img_sizes = [x.size + (i,) for i,x in enumerate(images)]


    scaled_width = int(in_scale * paper_width)
    scaled_height = int(in_scale * paper_height)
    # generates non-overlapping bounding boxes
    packer = newPacker(rotation=False, pack_algo=rectpack.GuillotineBafMaxas)
    packer.add_bin(scaled_width, scaled_height)

    for r in img_sizes:
        packer.add_rect(*r)
    packer.pack()
    
    paper = Image.new(color_channel, (scaled_width, scaled_height), color=(255,255,255,255)) # generate white rectangle
    
    if np.random.rand(1) < p_lines:
        # red line
        red_distance = int(1.25 * in_scale)
        red_line = Image.new(color_channel, (2, scaled_height), color=(255,0,0,255))
        paper.paste(red_line, (red_distance,0))

        # blue lines
        blue_height = (9 * in_scale) //32
        
    for rect in packer.rect_list():
        b, x, y, w, h, rid = rect

        bounding_boxes[rid] = (x, y, x+w, y+h)
        paper.paste(images[rid], (x,y), images[rid])
        
    return paper, bounding_boxes
