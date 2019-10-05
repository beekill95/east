import east.geometry as geometry
import numpy as np
from random import randint


def test_euclidean_distance():
    u = np.array([1, 0])
    v = np.array([1, 0])
    assert geometry.euclidean_distance(u, v) == 0


def test_minimum_bounding_box():
    box, angle = geometry.minimum_bounding_box(np.array([
        [1, 1],
        [2, 1],
        [3, 0],
        [0, 0]
    ]))

    assert np.allclose(box, np.array([
        [0, 1],
        [3, 1],
        [3, 0],
        [0, 0]
    ]))
    assert angle == 0


def test_polygon_clipping():
    crop_region = (1, 3, 1, 3)

    # Completely inside.
    points_inside = np.array([
        [1.5, 1.5],
        [1.5, 2.5],
        [2.5, 2.5],
        [2.5, 1.5]
    ])

    cropped_polygon = geometry.crop_polygon(points_inside, crop_region),
    assert np.allclose(cropped_polygon, points_inside)

    # Crop region is in the points.
    points_outside = np.array([
        [0, 0],
        [0, 4],
        [4, 4],
        [4, 0]
    ])

    cropped_polygon = geometry.crop_polygon(points_outside, crop_region),
    assert np.allclose(
        cropped_polygon,
        np.array([
            [1, 1],
            [1, 3],
            [3, 3],
            [3, 1]
        ])
    )

    # Part inside, part outside.
    points_in_out = np.array([
        [2, 2],
        [2, 4],
        [4, 4],
        [4, 2]
    ])

    cropped_polygon = geometry.crop_polygon(points_in_out, crop_region)
    assert np.allclose(
        cropped_polygon,
        np.array([
            [2, 2],
            [2, 3],
            [3, 3],
            [3, 2]
        ])
    )

    # Completely outside.
    points_completely_outside = np.array([
        [4, 4],
        [5, 4],
        [5, 5],
        [4, 5]
    ])
    cropped_polygon = geometry.crop_polygon(
        points_completely_outside, crop_region)
    assert len(cropped_polygon) == 0

    # TODO: we should write a test to test the case the box is square,
    # rotated, and have four corner outside of the crop region,
    # but majority of the box is inside.
    rotated_with_four_corners_outside_points = np.array([
        [2, 4],
        [4, 2],
        [2, 0],
        [0, 2]
    ])
    cropped_polygon = geometry.crop_polygon(
        rotated_with_four_corners_outside_points, crop_region)
    assert np.allclose(
        cropped_polygon,
        np.array([
            [3, 3],
            [3, 1],
            [1, 1],
            [1, 3]
        ])
    )

    # Edge case: there is a point on the border of crop region.
    points_on_crop_region_border = np.array([
        [3, 3],
        [3, 4],
        [4, 4],
        [4, 3.5]
    ])

    cropped_polygon = geometry.crop_polygon(
        points_on_crop_region_border, crop_region)
    assert len(cropped_polygon) == 1

    # Edge case: there is a line on the border of crop region.
    points_with_line_on_crop_region_border = np.array([
        [1, 3],
        [1, 5],
        [3, 5],
        [3, 3]
    ])
    cropped_polygon = geometry.crop_polygon(
        points_with_line_on_crop_region_border, crop_region)
    assert len(cropped_polygon) == 2


def test_polygon_clipping_should_not_produce_nan_or_inf_or_matching_vertices():
    crop_region = (200, 400, 200, 400)

    for i in range(100):
        polygon = np.array([
            [randint(150, 250), randint(150, 250)],
            [randint(150, 250), randint(350, 450)],
            [randint(350, 450), randint(350, 450)],
            [randint(350, 450), randint(150, 250)],
        ])

        cropped_polygon = geometry.crop_polygon(polygon, crop_region)

        uniques = np.unique(cropped_polygon, axis=0)
        assert len(uniques) == len(cropped_polygon)
        assert np.isfinite(cropped_polygon).all()
