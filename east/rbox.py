from east.geometry import euclidean_distance_point_line
import numpy as np


def generate_rbox(pixel, bounding_box):
    """
    Generate RBOX.

    :param pixel: location of the pixel, (x, y).
    :param bounding_box: a numpy array of size 4x2 coordinates of 4 vertices
    of the bounding box in clockwise order.
    :return a 1D numpy array of shape (4,) distance from current location
    to 4 edges in order (left, top, right, bottom).
    """
    assert len(bounding_box) == 4
    ordered_box = _reorder_vertices(bounding_box)
    edges = (ordered_box[i] - ordered_box[(i + 1) % 4] for i in range(4))
    distances = (euclidean_distance_point_line(pixel, edge) for edge in edges)
    return np.asarray(list(distances))


def decode_rbox(pixel, distances):
    """
    Decode RBOX.

    :param pixel: location of the pixel, (x, y)
    :param distances: distance of the pixel to 4 edges (left, top, right, bottom).
    :return: a numpy array of size 4x2 coordinates of 4 vertices of the bounding box
    in clockwise order, starting from top left corner.
    """
    x, y = pixel
    left_d, top_d, right_d, bottom_d = distances

    top = y - top_d
    bottom = y + bottom_d
    left = x - left_d
    right = x + right_d

    return np.array([
        [left, bottom]
        [left, top],
        [right, top],
        [right, bottom],
    ])


def _reorder_vertices(rectangular):
    """
    Reorder vertices in clock-wise order, with bottom-left vertices first.
    """
    s = rectangular[:, 0] + rectangular[:, 1]
    min_s_idx = np.argmin(s)
    return [rectangular[(min_s_idx + i) % 4] for i in range(4)]
