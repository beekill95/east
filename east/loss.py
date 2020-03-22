from tensorflow.python.keras import backend as K


def score_map_dice_loss(groundtruth_score_map, predicted_score_map, EPS=K.epsilon()):
    intersection = K.sum(groundtruth_score_map * predicted_score_map,
                         axis=[1, 2])
    union = (K.sum(groundtruth_score_map, axis=[1, 2])
             + K.sum(predicted_score_map, axis=[1, 2])
             + EPS)
    return 1. - (2. * intersection / union)


def score_map_dice_loss_log(groundtruth_score_map, predicted_score_map, EPS=K.epsilon()):
    intersection = K.sum(groundtruth_score_map * predicted_score_map,
                         axis=[1, 2])
    union = (K.sum(groundtruth_score_map, axis=[1, 2])
             + K.sum(predicted_score_map, axis=[1, 2])
             + EPS)
    return -K.log(EPS + intersection / union)


def score_map_loss(ground_truth_score_map, predicted_score_map, EPS=K.epsilon()):
    def log(x): return K.log(K.clip(x, EPS, 1.0))

    ground_truth_shape = K.cast(K.shape(ground_truth_score_map), 'float32')
    beta = 1 - (K.sum(ground_truth_score_map, axis=[1, 2], keepdims=True) /
                (ground_truth_shape[1] * ground_truth_shape[2]))

    loss = (- (beta * ground_truth_score_map * log(predicted_score_map))
            - ((1 - beta) * (1 - ground_truth_score_map) * log(1 - predicted_score_map)))

    return loss


def _rbox_angle_loss(ground_truth_angle, predicted_angle):
    return 1 - K.cos(predicted_angle - ground_truth_angle)


def _aabb_box_area(aabb):
    width = aabb[:, :, :, 1] + aabb[:, :, :, 3]
    height = aabb[:, :, :, 0] + aabb[:, :, :, 2]

    return width * height


def _aabb_intersected_area(aabb_1, aabb_2):
    min_distance = K.minimum(aabb_1, aabb_2)
    return _aabb_box_area(min_distance)


def _rbox_aabb_loss(ground_truth_aabb, predicted_aabb, EPS=K.epsilon()):
    ground_truth_area = _aabb_box_area(ground_truth_aabb)
    predicted_area = _aabb_box_area(predicted_aabb)

    intersected_area = _aabb_intersected_area(ground_truth_aabb,
                                              predicted_aabb)
    union_area = ground_truth_area + predicted_area - intersected_area

    # Equivalent to -log(intersected_area / union_area)
    return K.log(union_area + EPS) - K.log(intersected_area + EPS)


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
