from east import loss
from math import pi
import numpy as np
import pytest
from tensorflow.keras import backend as K


@pytest.fixture
def session():
    yield K.get_session()
    K.clear_session()


def test_score_map_loss(session):
    EPS = K.epsilon()
    def clip(x): return np.clip(x, EPS, 1.)

    # test with shape: (batch_size, w, h) = (1, 3, 4)
    gt = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    pred = np.array([
        [0, 0, 1, 0.5],
        [0, 0, 0.5, 0.5],
        [0, 0, 0.5, 0.5],
    ])
    gt_score_map = K.constant(np.array([gt]))

    pred_score_map = K.constant(np.array([pred]))

    l = loss.score_map_loss(gt_score_map, pred_score_map, EPS=EPS)
    l = l.eval(session=session)

    beta = 1. - 2/12
    expected_l = (-beta * gt * np.log(clip(pred))
                  - (1 - beta) * (1 - gt) * np.log(clip(1 - pred)))

    assert np.allclose(l, np.array([expected_l]))

    # test with shape: (batch_size, w, h) = (2, 3, 4)
    gt_2 = np.array([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ])
    pred_2 = np.array([
        [0, 0.75, 0.75, 0.5],
        [0.75, 0.5, 0.5, 0.25],
        [0, 0, 1, 0.25],
    ])

    gt_score_map = K.constant(np.array([gt, gt_2]))
    pred_score_map = K.constant(np.array([pred, pred_2]))

    l = loss.score_map_loss(gt_score_map, pred_score_map, EPS=EPS)
    l = l.eval(session=session)

    beta_2 = 1. - 5/12
    expected_l_2 = (-beta_2 * gt_2 * np.log(clip(pred_2))
                    - (1 - beta_2) * (1 - gt_2) * np.log(clip(1 - pred_2)))

    assert np.allclose(l, np.array([expected_l, expected_l_2]))


def test_angle_loss(session):
    gt_angle = K.constant(np.array([
        [[pi/6, pi/3], [pi/2, pi/4]],
        [[pi/4, pi/4], [pi/2, pi/2]]
    ]))
    pred_angle = K.constant(np.array([
        [[pi/3, pi/2], [pi/2, pi/2]],
        [[pi, pi/3], [pi/4, pi/2]]
    ]))

    l = loss._rbox_angle_loss(gt_angle, pred_angle)
    r = l.eval(session=session)

    assert np.allclose(r, 1 - np.cos(np.array([
        [[pi/6, -pi/6], [0, pi/4]],
        [[-3*pi/4, -pi/12], [pi/4, 0]]
    ])))


def test_aabb_box_area(session):
    left = np.array([[1.0, 2.0, 1.5], [0.5, 3.0, 2.0]])
    right = np.array([[1.5, 3.0, 2.5], [1.5, 0.7, 1.75]])
    top = np.array([[1.5, 3.0, 2.5], [1.5, 0.7, 1.75]])
    bottom = np.array([[1.0, 2.0, 1.5], [0.5, 3.0, 2.0]])

    rbox = np.asarray([left, top, right, bottom])
    rbox = np.moveaxis(rbox, 0, -1)
    rboxes = np.asarray([rbox, rbox])

    geometry_tensor = K.constant(rboxes)
    r = loss._aabb_box_area(geometry_tensor).eval(session=session)

    expected_area = np.array([
        [2.5*2.5, 5*5, 4*4],
        [2*2, 3.7*3.7, 3.75*3.75]
    ])

    assert np.allclose(r, np.asarray([
        expected_area,
        expected_area
    ]))


def test_aabb_intersected_area(session):
    pred_left = np.array([[1.0]])
    pred_right = np.array([[5.0]])
    pred_top = np.array([[9.0]])
    pred_bottom = np.array([[1.0]])

    gt_left = np.array([[2.0]])
    gt_right = np.array([[2.0]])
    gt_top = np.array([[2.0]])
    gt_bottom = np.array([[5.0]])

    pred_rbox = np.moveaxis(np.asarray([pred_left,
                                        pred_top,
                                        pred_right,
                                        pred_bottom]),
                            0,
                            -1)
    gt_rbox = np.moveaxis(np.asarray([gt_left,
                                      gt_top,
                                      gt_right,
                                      gt_bottom]),
                          0,
                          -1)

    pred_tensor = K.constant(np.asarray([pred_rbox]))
    gt_tensor = K.constant(np.asarray([gt_rbox]))

    r = (loss._aabb_intersected_area(pred_tensor,
                                     gt_tensor)
         .eval(session=session))

    assert np.shape(r) == (1, 1, 1)
    assert np.allclose(r, np.array([[[9]]]))
