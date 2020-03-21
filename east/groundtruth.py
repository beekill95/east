from east import geometry, rbox
from math import atan2
import numpy as np
from PIL import Image, ImageDraw


def _reorder_vertices(rectangular):
    """
    Reorder vertices in clock-wise order, with bottom-left vertices first.
    """
    s = rectangular[:, 0] + rectangular[:, 1]
    min_s_idx = np.argmin(s) - 1
    return np.asarray([rectangular[(min_s_idx + i) % 4]
                       for i in range(4)])


def _draw_shrinked_text_boxes(image, text_boxes, offset):
    img_height, img_width, _ = np.shape(image)

    shrinked_text_boxes = [geometry.shrink_polygon(box, offset)
                           for box in text_boxes]

    canvas = Image.new("L", (img_width, img_height))
    draw = ImageDraw.Draw(canvas)
    for i in range(len(text_boxes)):
        p = shrinked_text_boxes[i].flatten().tolist()
        draw.polygon(p, fill=(i+1))

    return np.asarray(canvas)[::4, ::4]


def _generate_score_map(shrinked_text_boxes_img):
    score_map = shrinked_text_boxes_img.astype(np.float32)
    score_map[score_map > 0] = 1
    return np.expand_dims(score_map, axis=0)


def _calculate_rotation_angle(rectangular):
    """
    Calculate the rotation angle of a rectangular, based on the top edge.
    |rectangular| is a 4x2 numpy array. Its vertices should be in clock-wise order,
    starting from bottom-left vertex.
    """
    edge = rectangular[2] - rectangular[1]
    return atan2(edge[1], edge[0])


def _generate_geometry_map(shrinked_text_boxes_img, text_boxes):
    # FIXME: handle the case when boxes are triangular, in that case,
    # minimum bounding boxes might be wrong.
    min_bboxes = (geometry.minimum_bounding_box(box) for box in text_boxes)
    min_bboxes = list(_reorder_vertices(box) for box, _ in min_bboxes)
    angles = list(_calculate_rotation_angle(box) for box in min_bboxes)

    img_shape = shrinked_text_boxes_img.shape
    geometry_map = np.zeros((5,) + img_shape)

    non_zero_rows, non_zero_cols = np.nonzero(shrinked_text_boxes_img)
    for i in range(len(non_zero_rows)):
        r, c = non_zero_rows[i], non_zero_cols[i]

        color = shrinked_text_boxes_img[r, c]
        bbox = min_bboxes[color - 1]

        # AABB & angle.
        geometry_map[:4, r, c] = rbox.generate_rbox((c * 4, r * 4),
                                                    bbox,
                                                    (img_shape[0] * 4, img_shape[1] * 4))
        geometry_map[4, r, c] = angles[color - 1]

    return geometry_map


def _generate_geometry_map_np(shrinked_text_boxes_img, text_boxes):
    # FIXME: handle the case when boxes are triangular, in that case,
    # minimum bounding boxes might be wrong.
    img_shape = shrinked_text_boxes_img.shape
    geometry_map = np.zeros((5,) + img_shape)
    bboxes = np.zeros((8,) + img_shape)

    for i in range(len(text_boxes)):
        min_bbox, _ = geometry.minimum_bounding_box(text_boxes[i])
        min_bbox = _reorder_vertices(min_bbox)

        mask = shrinked_text_boxes_img == i + 1

        # Assign bbox coordinates.
        # bboxes[mask] = min_bbox.flatten()
        # FIXME: don't use loop, use numpy broadcast to do the job.
        flatten_min_bbox = min_bbox.flatten()
        for i in range(8):
            bboxes[i][mask] = flatten_min_bbox[i]

        # Assign angles to geometry map.
        geometry_map[4][mask] = _calculate_rotation_angle(min_bbox)

    location_mask = shrinked_text_boxes_img > 0
    xv, yv = np.meshgrid(np.arange(0, img_shape[0]),
                         np.arange(0, img_shape[1]))
    locations = np.stack([xv, yv]) * 4

    geometry_map[:4] = rbox.generate_rbox_np(locations,
                                             bboxes,
                                             location_mask,
                                             (img_shape[0] * 4, img_shape[1] * 4))

    return geometry_map


def generate_ground_truth(image, text_boxes, score_map_offset=2):
    # FIXME: the offset should be changed accordingly to the size of the box,
    # or else the offsetted polygon would fall outside of the box.
    shrinked_text_boxes_img = _draw_shrinked_text_boxes(image,
                                                        text_boxes,
                                                        score_map_offset)

    # FIXME: we should filtered out boxes that are too small.
    score_map = _generate_score_map(shrinked_text_boxes_img)
    geometry_map = _generate_geometry_map_np(shrinked_text_boxes_img,
                                             text_boxes)
    gt_map = np.concatenate((score_map, geometry_map), axis=0)
    return np.moveaxis(gt_map, 0, -1)
