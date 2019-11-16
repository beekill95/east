"""
Process output tensor's of the network back to bounding boxes.
"""
from east import rbox, geometry
import numpy as np


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
