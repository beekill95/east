from east.geometry import euclidean_distance_vector_vector
import numpy as np


def generate_rbox_np(locations, bounding_boxes, mask):
    """
    Generate RBOX, but works on numpy arrays.

    :param locations: a 2xNxN numpy array stores the (x, y) coordinate of a pixel.
    :param bounding_boxes: a 8xNxN array stores the (x, y) coordinates of 4 vertices
    in clockwise order, starting from bottom left, of the bounding boxes.
    :param mask: a NxN numpy mask stores which location should be considered to generate rbox.
    :return: a 4xNxN numpy array store the distances of pixels in |locations| to corresponding
    edges in |bounding_boxes|. The distances are in order (left, top, right, bottom).
    """
    def dot(a, b):
        """
        Calculate the dot product of two 2xnxm numpy matrices, at a specific mask only.
        """
        product = np.multiply(a, b)
        return np.sum(product, axis=0)

    def euclidean_distance_np(locs, edges, out, mask):
        """
        Calculate the distance between 2xnxm locations with 2xnxm edges, at specific nxm mask only.
        """
        # Find the projection of the locs onto the edges.
        ratio = np.zeros(mask.shape)
        np.divide(dot(locs, edges), dot(edges, edges),
                  out=ratio, where=mask)
        projection = ratio * edges

        # Find the distance between the locs and its projections.
        b = locs - projection
        np.sqrt(dot(b, b), out=out, where=mask)

    edges = np.zeros((8,) + mask.shape)
    distances = np.zeros((4,) + mask.shape)

    # Edges.
    edges[:6] = bounding_boxes[:6] - bounding_boxes[2:]
    edges[6:8] = bounding_boxes[6:8] - bounding_boxes[:2]

    # Calculate the distances.
    for i in range(4):
        euclidean_distance_np(locations - bounding_boxes[i*2:(i+1)*2],
                              edges[i*2:(i+1)*2],
                              distances[i],
                              mask)

    return distances


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
    distances = (euclidean_distance_vector_vector(pixel - bounding_box[i],
                                                  bounding_box[i] - bounding_box[(i + 1) % 4])
                 for i in range(4))
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

    top = y + top_d
    bottom = y - bottom_d
    left = x - left_d
    right = x + right_d

    return np.array([
        [left, bottom],
        [left, top],
        [right, top],
        [right, bottom]
    ])
