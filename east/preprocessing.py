import east.geometry as geometry
import east.rbox as rbox
from functools import partial, reduce
from math import cos, sin, pi
import numpy as np
from PIL import Image, ImageDraw
import random
from tensorflow.python.keras.utils.data_utils import Sequence


def random_scale(scales, image, text_boxes):
    """
    Random scale an image and its boxes.

    :param scales: a list stores random scales to be chosen.
    :param image: image to be scaled.
    :param text_boxes: a list of text boxes. Each box is a numpy array of size nx2.
    :return: a tuple of scaled image and text boxes.
    """
    scale = random.choice(scales)
    img_w, img_h = image.size

    scaled_img = image.resize((int(img_w * scale), int(img_h * scale)))
    scaled_text_boxes = [box * scale for box in text_boxes]

    return scaled_img, scaled_text_boxes


def random_rotate(angles, image, text_boxes):
    """
    Random rotate an image and its boxes.

    :param angles: a tuple (min, max) indicates the range of the random angle in degree.
    :param image: image to be rotated.
    :param text_boxes: a list of text boxes. Each box is a numpy array of size nx2.
    :return: a tuple of rotated image and text boxes.
    """
    angle = random.randint(angles[0], angles[1]) * pi / 180

    rotated_img = image.rotate(angle)

    # Rotate text boxes.
    img_center = np.asarray(image.size) / 2
    rotated_text_boxes = [
        geometry.rotate_polygon(box, angle, img_center) for box in text_boxes]

    return rotated_img, rotated_text_boxes


def random_crop_with_text_boxes_cropped(target_size, at_least_one_box_ratio, image, text_boxes):
    """
    Random crop an image, also crop the boxes.
    If the image size is smaller than target size,
    no operation is done and original image and boxes will be return.

    :param target_size: target size of output image. This is a tuple of (width, height).
    :param at_least_one_box_ratio: control whether the crop algorithm should try to crop a region
    in image that will contain a text box. The value must be in range [0, 1].
    :param image: a Pillow image to be cropped.
    :param text_boxes: a list of text boxes. Each box is a numpy array of size nx2.
    If this is False, the function will try to find the best region to crop to contain at least a text box.
    :return: a cropped image and boxes.
    """
    def find_good_crop_start(img_width, img_height, target_width, target_height, ensure_at_least_one_box):
        if not ensure_at_least_one_box:
            crop_left = (0 if img_width <= target_width
                         else random.randint(0, img_width - target_width - 1))
            crop_top = (0 if img_height <= target_height
                        else random.randint(0, img_height - target_height - 1))
            return crop_left, crop_top
        else:
            # Idea:
            # We will use row indicators and col indicators to mark which region
            # in the image has a text box.
            # To indicate box ith separately, we will use value 2^i to do that.
            # Each indicator will has a value, indicate which boxes in that spot.
            # For example, the value of 32 indicates the box 5th is in the spot.
            # Or 34 indicates box 1st and box 5th is in the spot.
            # Then we randomly choose the indicator in the row indicators.
            # Then, based on the value of the choosen row indicator, we filtered
            # out column indicators, and then randomly choose the value on
            # that indicators.
            x_box_region = np.zeros(img_width, dtype=np.uint32)
            y_box_region = np.zeros(img_height, dtype=np.uint32)

            # Limit number of text boxes to maximum of 32 because
            # the algorithm is limited by int32.
            # In that case, randomly shuffle all box indices and
            # select the first 32. That way, every box has an equal
            # change to appear in the cropped region.
            num_text_boxes = len(text_boxes)
            box_indices = (np.arange(num_text_boxes) if num_text_boxes < 32
                           else np.random.permutation(num_text_boxes)[:32])

            for i in range(len(box_indices)):
                text_box = text_boxes[box_indices[i]]

                min_x, max_x = np.min(text_box[:, 0]), np.max(text_box[:, 0])
                min_y, max_y = np.min(text_box[:, 1]), np.max(text_box[:, 1])

                start_x = max(int(min_x - target_width / 2), 0)
                end_x = int((min_x + max_x) / 2)
                x_box_region[start_x:end_x] |= 1 << i

                start_y = max(int(min_y - target_height / 2), 0)
                end_y = int((min_y + max_y) / 2)
                y_box_region[start_y:end_y] |= 1 << i

            # FIXME: this selection process will be bias toward
            # long and tall boxes.
            good_x = np.nonzero(x_box_region)[0]
            chosen_x = np.random.choice(good_x)
            chosen_x_val = x_box_region[chosen_x]

            good_y = np.nonzero(y_box_region & chosen_x_val)[0]
            chosen_y = np.random.choice(good_y)

            return chosen_x, chosen_y

    img_width, img_height = image.size
    target_width, target_height = target_size

    if img_width <= target_width and img_height <= target_height:
        return image, text_boxes

    ensure_at_least_one_box = (len(text_boxes)
                               and random.uniform(0, 1) <= at_least_one_box_ratio)
    crop_left, crop_top = find_good_crop_start(img_width, img_height,
                                               target_width, target_height,
                                               ensure_at_least_one_box)

    cropped_img = image.crop((crop_left,
                              crop_top,
                              crop_left + target_width,
                              crop_top + target_height))

    cropped_text_boxes = (geometry.crop_polygon(
        box,
        (crop_left, crop_left + target_width,
         crop_top, crop_top + target_height)) for box in text_boxes)
    # Filtered out invalid boxes: boxes must have at least 3 vertices.
    cropped_text_boxes = filter(lambda b: len(b) >= 3, cropped_text_boxes)

    return cropped_img, list(cropped_text_boxes)


def pad_image(target_size, image, text_boxes):
    img_width, img_height = image.size
    target_width, target_height = target_size

    if img_width >= target_width and img_height >= target_height:
        return image, text_boxes

    padded_img = Image.new(image.mode, target_size)
    padded_img.paste(image)

    return padded_img, text_boxes


def generate_ground_truth(image, text_boxes, score_map_offset=0.3):
    def draw_shrinked_text_boxes(offset):
        img_height, img_width, _ = np.shape(image)

        shrinked_text_boxes = [
            geometry.shrink_polygon(box, offset) for box in text_boxes]

        canvas = Image.new("L", (img_width, img_height))
        draw = ImageDraw.Draw(canvas)
        for i in range(len(text_boxes)):
            draw.polygon(shrinked_text_boxes[i].flatten(), fill=(i+1))

        return np.asarray(canvas)

    def generate_score_map(shrinked_text_boxes_img):
        score_map = shrinked_text_boxes_img.astype(np.float32)
        score_map[score_map > 0] = 1
        score_map = score_map[::4, ::4]
        return np.expand_dims(score_map, axis=0)

    def generate_geometry_map(shrinked_text_boxes_img):
        min_bboxes = [geometry.minimum_bounding_box(box) for box in text_boxes]

        img_height, img_width, _ = np.shape(image)
        geometry_map = np.zeros((5, img_height, img_width))

        non_zero_x, non_zero_y = np.nonzero(shrinked_text_boxes_img)
        for i in range(len(non_zero_x)):
            r, c = non_zero_x[i], non_zero_y[i]

            color = shrinked_text_boxes_img[r, c]
            bbox = min_bboxes[color - 1]

            # AABB & angle.
            geometry_map[:4, r, c] = rbox.generate_rbox((r, c), bbox[0])
            geometry_map[4, r, c] = bbox[1]

        return geometry_map[:, ::4, ::4]

    shrinked_text_boxes_img = draw_shrinked_text_boxes(score_map_offset)

    score_map = generate_score_map(shrinked_text_boxes_img)
    geometry_map = generate_geometry_map(shrinked_text_boxes_img)
    gt_map = np.concatenate((score_map, geometry_map), axis=0)
    return np.moveaxis(gt_map, 0, -1)


def process_data(pipeline, image, text_boxes):
    pil_img = Image.fromarray(image)
    np_text_boxes = [np.asarray(box[1:]).reshape(-1, 2)
                     for box in text_boxes]

    processed_img, processed_text_boxes = reduce(
        lambda x, f: f(image=x[0], text_boxes=x[1]),
        pipeline,
        (pil_img, np_text_boxes))

    return np.asarray(processed_img), processed_text_boxes


def flow_from_generator(data_iter, processing_pipeline):
    """
    Convert data loaded from dataset into input images X and ground truth Y.

    :param data_iter: iterable of dataset, each call to next will return a list of
    tuples of (numpy images, list_of_boxes_in_images). Each box geometry is a list of
    [difficulty, x1, y1, x2, y2, x3, y3, x4, y4].
    :param processing_pipeline: a list of partial functions to preprocess images and text boxes.
    Each function should expected to accept to kwargs: image and text_boxes, it should return a
    tuple of two (image, text_boxes).
    :return: a generator yields batch of images and groundtruths.
    """
    process_data_fn = partial(process_data, processing_pipeline)

    for data in data_iter:
        images = []
        gts = []

        processed_data = (process_data_fn(d[0], d[1])
                          for d in zip(data[0], data[1]))

        for d in processed_data:
            gt = generate_ground_truth(d[0], d[1])

            images.append(d[0])
            gts.append(gt)

        yield np.asarray(images), np.asarray(gts)


class PreprocessingSequence(Sequence):
    """
    Preprocessing input images and groundtruth text boxes
    into expected format by the model using Keras Sequence API.
    """

    def __init__(self, data_sequence, preprocessing_pipeline):
        """
        Initialize preprocessing sequence.

        :param data_sequence: a Keras Sequence produces a batch of images and groundtruth text boxes.
        Expected each image to be a numpy array, and each text box is a list of [difficulty, x1, y1, x2, y2, x3, y3, x4, y4].
        :param preprocessing_pipeline: a list of preprocess functions to be applied to an image and its text boxes.
        Those functions should expected to receive two keyword arguments: image and text_boxes.
        """
        self._data_sequence = data_sequence
        self._preprocessing_fn = partial(process_data, preprocessing_pipeline)

    def __len__(self):
        return len(self._data_sequence)

    def __getitem__(self, index):
        images, groundtruths = self._data_sequence[index]

        preprocessed_images = []
        preprocessed_gts = []

        for i in range(len(images)):
            r = self._preprocessing_fn(images[i], groundtruths[i])
            gt = generate_ground_truth(r[0], r[1])

            preprocessed_images.append(r[0])
            preprocessed_gts.append(gt)

        return np.asarray(preprocessed_images), np.asarray(preprocessed_gts)

    def on_epoch_end(self):
        self._data_sequence.on_epoch_end()
