from dataset.train_validation_splitter import TrainValidationSplitter
import dataset.utils as data_utils
from east.geometry import rotate_polygon
import math
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from PIL import Image
from random import shuffle
from tensorflow.python.keras.utils.data_utils import Sequence


def load_msra_td_500_generator(path, batch_size, shuffle=True):
    return iter(MSRA(path, batch_size, shuffle))


class MSRA:
    """
    Load the data in MSRA data set in an iterable.
    """

    def __init__(self, data_set_path, batch_size, shuffle=True):
        self._image_files = data_utils.list_all_images(data_set_path)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._data_set_path = data_set_path
        self._max_steps = int(len(self._image_files) / self._batch_size)

    @property
    def steps_per_epoch(self):
        return self._max_steps

    def __iter__(self):
        self._step = 0
        return self

    def __next__(self):
        batch_size = self._batch_size
        max_steps = self._max_steps

        if self._step == max_steps:
            self._step = 0
            if self._shuffle:
                shuffle(self._image_files)

        next_index = ((self._step + 1) * batch_size
                      if self._step + 1 < max_steps
                      else len(self._image_files))

        images = []
        gts = []
        for i in range(self._step * batch_size, next_index):
            img_file = self._image_files[i]
            img_path = _full_path(self._data_set_path, img_file)
            gt_path = _full_path(self._data_set_path, _gt_file(img_file))

            gt = _load_gt(gt_path)
            box_coordinates = [g[:1] + _convert_geometry_to_coordinates(g[1:])
                               for g in gt]

            images.append(_load_img(img_path))
            gts.append(box_coordinates)

        self._step += 1

        return images, gts

    def _load_gt(self, img_file):
        """
        Load ground truth textboxes into a list.
        Each item in list will be another list:
        [difficulty, x, y, width, height, angle]
        """
        gt_file = _gt_file(img_file)
        gt_path = _full_path(self._data_set_path, gt_file)
        return _load_gt(gt_path)


class MSRASequence(Sequence):
    """
    Load MSRA data set in Keras Sequence API.
    """

    def __init__(self, msra_data_path, batch_size, shuffle=True, multithread=None):
        self._msra_path = msra_data_path
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._multithread = multithread
        self._image_paths = data_utils.list_all_images(msra_data_path)

    def __len__(self):
        size = len(self._image_paths) / self._batch_size
        return int(math.ceil(size))

    def __getitem__(self, index):
        batch_size = self._batch_size
        max_steps = len(self)

        next_index = ((index + 1) * batch_size
                      if index + 1 < max_steps
                      else len(self._image_paths))

        img_files = self._image_paths[index * batch_size:next_index]

        def load_img_gt(img_file):
            img_path = _full_path(self._msra_path, img_file)
            gt_path = _full_path(self._msra_path, _gt_file(img_file))

            gt = _load_gt(gt_path)
            box_coordinates = [g[:1] + _convert_geometry_to_coordinates(g[1:])
                               for g in gt]

            return _load_img(img_path), box_coordinates

        if self._multithread:
            thread_pool = ThreadPool(self._multithread)
            results = thread_pool.map(load_img_gt, img_files)
        else:
            results = [load_img_gt(f) for f in img_files]

        return list(zip(*results))

    def on_epoch_end(self):
        if self._shuffle:
            shuffle(self._image_paths)


class MSRATrainValidationSplitter(TrainValidationSplitter):
    def __init__(self, msra_path, validation_percentage):
        super().__init__(validation_percentage)
        self._msra_path = data_utils.absolute_path(msra_path)

    def _split_training_data(self, train_dir, val_dir, validation_percentage):
        image_names = data_utils.list_all_images(self._msra_path)

        total_images = len(image_names)
        nb_train_images = int(total_images * (1 - validation_percentage))

        for i in range(total_images):
            img_name = image_names[i]
            gt_name = _gt_file(img_name)

            link_dir = train_dir if i <= nb_train_images else val_dir

            # Create symbolic link in the link directory.
            data_utils.symlink(data_utils.join_path(link_dir, img_name),
                               data_utils.join_path(self._msra_path, img_name))
            data_utils.symlink(data_utils.join_path(link_dir, gt_name),
                               data_utils.join_path(self._msra_path, gt_name))


def _gt_file(img_file):
    img_name = data_utils.get_file_name(img_file, False)
    return f'{img_name}.gt'


def _full_path(msra_path, file_name):
    return data_utils.join_path(msra_path, file_name)


def _load_img(img_path):
    return np.asarray(Image.open(img_path))


def _load_gt(gt_path):
    text_boxes = []

    with open(gt_path, 'r') as f:
        for line in f.readlines():
            box_encoded = line.rstrip().split(' ')
            angle = float(box_encoded[-1])
            box = [int(n) for n in box_encoded[1:-1]] + [angle]
            text_boxes.append(box)

    return text_boxes


def _convert_geometry_to_coordinates(box_geometry):
    """
    Convert box geometry to box coordinates.
    Box geometry is a list of [x, y, width, height, angle].
    Return coordinates of box's vertices: [x1, y1, ..., x4, y4].
    """
    x, y, width, height, angle = box_geometry
    center_x, center_y = x + width / 2, y + height / 2

    points = np.array([
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height]
    ])
    rotated_points = rotate_polygon(points, angle, (center_x, center_y))
    return rotated_points.flatten().astype(np.int32).tolist()
