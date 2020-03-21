import tensorflow as tf
from tensorflow.python.keras import backend as K
# FIXME: many of these are using tensorflow API directly. We should convert it to use Keras API.


def score_map_dice_loss(groundtruth_score_map, predicted_score_map, EPS=K.epsilon()):
    intersection = K.sum(groundtruth_score_map * predicted_score_map,
                         axis=[1, 2])
    union = (K.sum(groundtruth_score_map, axis=[1, 2])
             + K.sum(predicted_score_map, axis=[1, 2])
             + EPS)
    return 1. - (2. * intersection / union)


def score_map_dice_loss_all(groundtruth, predicted, EPS=K.epsilon()):
    intersection = K.sum(groundtruth * predicted, axis=[1, 2, 3])
    union = (K.sum(groundtruth, axis=[1, 2, 3])
             + K.sum(predicted, axis=[1, 2, 3])
             + EPS)
    return 1. - (2. * intersection / union)


def score_map_loss(ground_truth_score_map, predicted_score_map, EPS=K.epsilon()):
    def log(x): return tf.log(tf.clip_by_value(x, EPS, 1.0))

    ground_truth_shape = tf.cast(tf.shape(ground_truth_score_map), tf.float32)
    beta = 1 - (tf.math.reduce_sum(ground_truth_score_map, axis=[1, 2], keep_dims=True) /
                (ground_truth_shape[1] * ground_truth_shape[2]))

    loss = (- (beta * ground_truth_score_map * log(predicted_score_map))
            - ((1 - beta) * (1 - ground_truth_score_map) * log(1 - predicted_score_map)))

    return loss


def score_map_balanced_entropy_all(ground_truth_score_map, predicted_score_map, EPS=K.epsilon()):
    def log(x): return K.log(x + EPS)

    ground_truth_shape = tf.cast(tf.shape(ground_truth_score_map), tf.float32)
    beta = 1 - (K.sum(ground_truth_score_map) / K.prod(ground_truth_shape))

    loss = (- (beta * ground_truth_score_map * log(predicted_score_map))
            - ((1 - beta) * (1 - ground_truth_score_map) * log(1 - predicted_score_map)))

    return K.sum(loss)


def _rbox_angle_loss(ground_truth_angle, predicted_angle):
    return 1 - tf.cos(predicted_angle - ground_truth_angle)


def _aabb_box_area(aabb):
    width = aabb[:, :, :, 1] + aabb[:, :, :, 3]
    height = aabb[:, :, :, 0] + aabb[:, :, :, 2]

    return width * height


def _aabb_intersected_area(aabb_1, aabb_2):
    min_distance = tf.math.minimum(aabb_1, aabb_2)
    return _aabb_box_area(min_distance)


def _rbox_aabb_loss(ground_truth_aabb, predicted_aabb, EPS=K.epsilon()):
    ground_truth_area = _aabb_box_area(ground_truth_aabb)
    predicted_area = _aabb_box_area(predicted_aabb)

    intersected_area = _aabb_intersected_area(ground_truth_aabb,
                                              predicted_aabb)
    union_area = ground_truth_area + predicted_area - intersected_area

    return -tf.math.log((intersected_area + EPS) / (union_area + EPS))


def rbox_geometry_loss(ground_truth_rbox_geometry, predicted_rbox_geometry, lambda_term=10, EPS=K.epsilon()):
    # AABB loss.
    ground_truth_aabb = ground_truth_rbox_geometry[:, :, :, :4]
    predicted_aabb = predicted_rbox_geometry[:, :, :, :4]
    rbox_aabb_loss = _rbox_aabb_loss(ground_truth_aabb,
                                     predicted_aabb,
                                     EPS=EPS)

    # Angle loss.
    ground_truth_angle = ground_truth_rbox_geometry[:, :, :, 4]
    predicted_angle = predicted_rbox_geometry[:, :, :, 4]
    angle_loss = _rbox_angle_loss(ground_truth_angle, predicted_angle)

    return rbox_aabb_loss + lambda_term * angle_loss


def rbox_geometry_loss_with_beta(groundtruth, prediction, lambda_term=1, EPS=K.epsilon()):
    ground_truth_score_map = groundtruth[:, :, :, 0]
    ground_truth_shape = K.cast(K.shape(ground_truth_score_map), 'float32')
    beta = (K.sum(ground_truth_score_map, axis=[1, 2], keepdims=True) /
            (ground_truth_shape[1] * ground_truth_shape[2]))

    ground_truth_rbox_geometry = groundtruth[:, :, :, 1:]
    predicted_rbox_geometry = prediction[:, :, :, 1:]

    # AABB loss.
    ground_truth_aabb = ground_truth_rbox_geometry[:, :, :, :4]
    predicted_aabb = predicted_rbox_geometry[:, :, :, :4]
    rbox_aabb_loss = _rbox_aabb_loss(ground_truth_aabb,
                                     predicted_aabb,
                                     EPS=EPS)

    # Angle loss.
    ground_truth_angle = ground_truth_rbox_geometry[:, :, :, 4]
    predicted_angle = predicted_rbox_geometry[:, :, :, 4]
    angle_loss = _rbox_angle_loss(ground_truth_angle, predicted_angle)

    return (rbox_aabb_loss + lambda_term * angle_loss) / (beta + EPS)
