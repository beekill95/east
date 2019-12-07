from math import sqrt, cos, sin, isclose
import numpy as np
from scipy.spatial import ConvexHull


def magnitude(v):
    """Calculate magnitude of a numpy vector"""
    return sqrt(np.dot(v, v))


def euclidean_distance(u, v):
    """Calculate Euclidean distance between two numpy 1D vectors"""
    return magnitude(u - v)


def euclidean_distance_vector_vector(v, u):
    projection = np.dot(v, u) / np.dot(u, u) * u
    return euclidean_distance(v, projection)


def rotate_polygon(polygon, angle, anchor=None):
    """
    Rotate a polygon around the anchor.

    :param polygon: a numpy matrix of size nx2 contains vertices of polygon.
    :param angle: rotation angle, in radian.
    :param anchor: an anchor to rotate the polygon around. If the anchor is None,
    it is rotated around the angle.
    :return: return a numpy matrix of size nx2 of vertices of rotated polygon.
    """
    rotation_matrix_transposed = np.array([
        [cos(angle), sin(angle)],
        [-sin(angle), cos(angle)]
    ])

    anchor = np.array([0, 0]) if anchor is None else anchor
    return (polygon - anchor) @ rotation_matrix_transposed + anchor


def minimum_bounding_box(points):
    """
    Calculate minimum area enclosing box covers |points|.

    :param points: a numpy matrix of nx2 coordinates of points.
    :return: a tuple of:
      * a numpy matrix of nx2 coordinates of 4 vertices of rotated enclosing box in clockwise order.
      * rotation angle.
    """
    hull_vertices = points[ConvexHull(points).vertices]

    # Calculate edges of the convex hull.
    hull_edges = np.zeros(hull_vertices.shape)
    hull_edges[:-1] = hull_vertices[1:] - hull_vertices[:-1]
    hull_edges[-1] = hull_vertices[0] - hull_vertices[-1]

    # Angles of each edge.
    hull_angles = np.arctan2(hull_edges[:, 1], hull_edges[:, 0])
    hull_angles = np.unique(hull_angles)

    # Generate rotation matrix for each edge angle to
    # have an edge parallel to horizontal axis.
    rotation_matrix = np.vstack((
        np.cos(-hull_angles),
        -np.sin(-hull_angles),
        np.sin(-hull_angles),
        np.cos(-hull_angles)
    ))
    rotation_matrix = rotation_matrix.T.reshape(-1, 2, 2)

    rotated_vertices = rotation_matrix @ hull_vertices.T

    min_x = np.min(rotated_vertices[:, 0], axis=1)
    max_x = np.max(rotated_vertices[:, 0], axis=1)
    min_y = np.min(rotated_vertices[:, 1], axis=1)
    max_y = np.max(rotated_vertices[:, 1], axis=1)

    areas = (max_x - min_x) * (max_y - min_y)
    best_area_idx = np.argmin(areas)

    best_min_x = min_x[best_area_idx]
    best_max_x = max_x[best_area_idx]
    best_min_y = min_y[best_area_idx]
    best_max_y = max_y[best_area_idx]
    best_angle = hull_angles[best_area_idx]

    # TODO: this can be further optimized by reusing
    # the previous rotation_matrix.
    rotate_back_matrix = np.asarray([
        [cos(best_angle), -sin(best_angle)],
        [sin(best_angle), cos(best_angle)]
    ])

    # Rotate the points back to original orientation.
    min_rect = np.asarray([
        [best_min_x, best_min_y],
        [best_max_x, best_min_y],
        [best_max_x, best_max_y],
        [best_min_x, best_max_y],
    ]) @ rotate_back_matrix.T

    return min_rect, best_angle


def crop_polygon(points, crop_region):
    """
    Crop the polygon by the crop region using Sutherland-Hodgman algorithm.

    :param points: a numpy matrix of nx2 coordinates of points.
    :param crop_region: a tuple of (min_x, max_x, min_y, max_y) of the crop region.
    :return: a numpy matrix of nx2 coordinates of cropped polygon.
    """
    def is_on_line(point, line):
        x, y = point
        x_line, y_line = line
        return ((x_line is not None and isclose(x, x_line))
                or (y_line is not None and isclose(y, y_line)))

    def is_on_left_side(point, line):
        x, y = point
        x_line, y_line = line
        return ((x_line is not None and x < x_line)
                or (y_line is not None and y < y_line))

    def intersect_with_line(in_point, out_point, line):
        x_line, y_line = line
        x_in, y_in = in_point
        x_out, y_out = out_point

        # (x - x_in) / (y - y_in) = (x_out - x_in) / (y_out - y_in)
        # FIXME: y_out - y_in could be zero. Similarly, in_out_slope could be zero.
        in_out_slope = (x_out - x_in) / (y_out - y_in)
        if x_line == None:
            x = x_in + (y_line - y_in) * in_out_slope
            return x, y_line
        else:
            y = y_in + (x_line - x_in) / in_out_slope
            return x_line, y

    def crop_with(points_iter, line, keep_left):
        """
        Crop the points with a line, keep the points that is in the left side of the line,
        if |keep_left| is True, and the right side otherwise.

        :param points_iter: iterator of each vertex of the polygon.
        :param line: a tuple of (x, y) indicate the line to crop the polygon with.
        :param keep_left: to keep the vertices on the left or right side of the line.
        :return: an iterator of all vertices cropped with the line.
        """
        previous_point = None
        is_previous_point_inside = None
        first_inside_point = None

        def set_first_point(point):
            nonlocal first_inside_point
            if first_inside_point is None:
                first_inside_point = point

        for point in points_iter:
            is_left_side = is_on_left_side(point, line)
            is_inline = is_on_line(point, line)
            is_inside = (is_left_side == keep_left) or is_inline

            if (previous_point is not None
                    and is_previous_point_inside != is_inside
                    and not is_inline
                    and not is_on_line(previous_point, line)):
                intersection = intersect_with_line(point, previous_point, line)

                set_first_point(intersection)
                yield intersection

            previous_point = point
            is_previous_point_inside = is_inside

            if is_inside:
                set_first_point(point)
                yield point

        if (first_inside_point is not None
                and not np.array_equiv(first_inside_point, previous_point)):
            yield first_inside_point

    min_x, max_x, min_y, max_y = crop_region

    clipped_polygon = (points[i % len(points)] for i in range(len(points) + 1))
    clipped_polygon = crop_with(clipped_polygon, (min_x, None), False)
    clipped_polygon = crop_with(clipped_polygon, (None, max_y), True)
    clipped_polygon = crop_with(clipped_polygon, (max_x, None), True)
    clipped_polygon = crop_with(clipped_polygon, (None, min_y), False)

    clipped_polygon = np.asarray(list(clipped_polygon))
    return clipped_polygon[:-1]


def shrink_polygon(polygon, offset):
    """
    FIXME: does the polygon has to be in clock-wise or counter clock-wise order?
    Offset polygon according to offset.
    Ported from Mr. Rafsanjani's comments in
    [Pyright](http://pyright.blogspot.com/2014/11/polygon-offset-with-pyeuclid-revisited.html). 

    :param polygon: a nx2 numpy matrix contains all vertices of the polygon.
    :param offset: ratio of the shrink.
    :return: a nx2 numpy matrix of the vertices of the shrinked polygon.
    """
    assert offset > 0, "Only accept positive value for offset"

    def offset_corner(point_1, point_2, point_3):
        """
        Offset a polygon's corner specified by 3 points.
        All the points are numpy arrays.
        Return an offsetted point from |point_2| by an offset.
        """
        v21_norm = (point_1 - point_2) / magnitude(point_1 - point_2)
        v23_norm = (point_3 - point_2) / magnitude(point_3 - point_2)

        # Bisector of the corner formed by v21 and v23.
        bisector = v21_norm + v23_norm
        bisector = bisector / magnitude(bisector)

        # e = bisector - projection of bisector onto v23_norm
        e = bisector - np.dot(bisector, v23_norm) * v23_norm

        # Calculate the cross product of v21 and v23 to determine
        # in which direction should we offset the corder to ensure
        # the inset corner is inside the polygon.
        # TODO: check if the sign is correct.
        sign = 1 if np.cross(v21_norm, v23_norm) <= 0 else -1
        return point_2 + sign * offset * magnitude(e) * bisector

    inset_polygon = np.zeros(polygon.shape)
    num_vertices = len(polygon)
    for i in range(num_vertices):
        middle_idx = (i + 1) % num_vertices
        last_idx = (i + 2) % num_vertices
        inset_polygon[middle_idx] = offset_corner(polygon[i],
                                                  polygon[middle_idx],
                                                  polygon[last_idx])

    return inset_polygon
