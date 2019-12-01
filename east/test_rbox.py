from east import rbox, geometry
from math import sqrt, pi
import numpy as np


def test_encode_rbox():
    box = rbox.generate_rbox((8, 8),
                             np.array([
                                 [4, 16],
                                 [4, 4],
                                 [16, 4],
                                 [16, 16]
                             ]),
                             (16, 16))

    assert np.allclose(box, np.array([0.25, 0.25, 0.5, 0.5]))

    box = rbox.generate_rbox((8, 8),
                             np.array([
                                 [0, 16],
                                 [0, 4],
                                 [16, 4],
                                 [16, 16]
                             ]),
                             (16, 16))

    assert np.allclose(box, np.array([0.5, 0.25, 0.5, 0.5]))

    box = rbox.generate_rbox((8, 8),
                             np.array([
                                 [0, 8],
                                 [8, 0],
                                 [16, 8],
                                 [8, 16]
                             ]),
                             (16, 16))
    assert np.allclose(box, np.array((0.25 * sqrt(2), ) * 4))


def test_decode_rbox():
    box = rbox.decode_rbox((8, 8), (4/16, 4/16, 8/16, 8/16), (16, 16))
    assert np.allclose(box, np.array([[4, 16], [4, 4], [16, 4], [16, 16]]))

    box = rbox.decode_rbox((8, 8), np.array((0.25 * sqrt(2), ) * 4), (16, 16))
    assert np.allclose(box, np.array([
        [8 - 4 * sqrt(2), 8 + 4 * sqrt(2)],
        [8 - 4 * sqrt(2), 8 - 4 * sqrt(2)],
        [8 + 4 * sqrt(2), 8 - 4 * sqrt(2)],
        [8 + 4 * sqrt(2), 8 + 4 * sqrt(2)],
    ]))
