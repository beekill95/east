"""
Process output tensor's of the network back to bounding boxes.
"""
from east import rbox, geometry
import numpy as np
from PIL import Image, ImageDraw


def extract_text_boxes(predicted, orig_img_size, score_threshold=0.5):
    """
    Extract text boxes from output tensor of the network.

    :param predicted: network's output tensor of size nxmx6.
    :param score_threshold: threshold to consider as a valid box.
    :return: a list of text boxes' coordinates and its score: a list of
    (score, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).
    """
    score_map = predicted[:, :, 0]
    box_geometry = predicted[:, :, 1:5]
    angle = predicted[:, :, 5]

    rows, cols = np.where(score_map >= score_threshold)
    boxes = []
    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        box = rbox.decode_rbox((c * 4, r * 4),
                               box_geometry[r, c, :],
                               orig_img_size)
        rotated_box = geometry.rotate_polygon(box,
                                              angle[r, c],
                                              (c * 4, r * 4))

        boxes.append((score_map[r, c], rotated_box))

    return boxes


def filter_text_boxes(score_map, predicted_text_boxes, image_size, box_score_threshold=0.1):
    """
    Filter out predicted text boxes that only cover parts of the score map.

    :param score_map: predicted score map from the network.
    :param predicted_text_boxes: a list of text boxes' coordinates and its score:
    (score, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).
    :param image_size: (w, h) of the image.
    :param box_score_threshold: threshold that the text boxes must pass to be considered as valid box.
    :return: a list contains good text boxes.
    """
    good_boxes = []

    for box in predicted_text_boxes:
        coords = box[1]

        # Draw the box in the original image size.
        img = Image.new('L', image_size)
        draw = ImageDraw.Draw(img)
        draw.polygon(coords.flatten().tolist(), fill=(1))

        # Calculate the box score.
        box_score_map = np.asarray(img)[::4, ::4]
        area = np.sum(box_score_map)
        box_score = np.sum(score_map * box_score_map) / area

        # Filter out bad box.
        if box_score >= box_score_threshold:
            good_boxes.append(box)

    return good_boxes
