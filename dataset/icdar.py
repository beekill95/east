from dataset.train_validation_splitter import TrainValidationSplitter
from dataset import path
from math import ceil
import numpy as np
from PIL import Image
from random import shuffle
from tensorflow.python.keras.utils.data_utils import Sequence


class ICDAR2015Sequence(Sequence):
    """
    Load ICDAR 2015 Robust Reading challenge.
    """

    def __init__(self, icdar_2105_data_path, batch_size, shuffle=True):
        self._icdar_path = icdar_2105_data_path
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._image_paths = path.list_all_images(self._icdar_path)

    def __len__(self):
        size = len(self._image_paths) / self._batch_size
        return int(ceil(size))

    def __getitem__(self, index):
        batch_size = self._batch_size
        max_steps = len(self)

        next_index = ((index + 1) * batch_size
                      if index + 1 < max_steps
                      else len(self._image_paths))

        img_files = self._image_paths[index * batch_size:next_index]

        def load_gt(gt_path):
            text_boxes = []
            with open(gt_path, 'r', encoding='utf-8-sig') as file:
                for line in file:
                    # Get the first 8 box coordinates and convert to location.
                    fields = line.split(',')[:8]
                    coords = [int(c) for c in fields[:8]]

                    text = ','.join(fields[8:])

                    # Append 0 for dummy difficulty.
                    if text != '###':
                        text_boxes.append([0, *coords])

            return text_boxes

        def load_image(img_path):
            return np.asarray(Image.open(img_path))

        def load_image_gt(image_file):
            img_path = path.join_path(self._icdar_path, image_file)
            gt_path = path.join_path(self._icdar_path,
                                     _gt_file(image_file))

            return load_image(img_path), load_gt(gt_path)

        results = (load_image_gt(f) for f in img_files)
        return list(zip(*results))

    def on_epoch_end(self):
        if self._shuffle:
            shuffle(self._image_paths)


class ICDAR2015TrainValidationSplitter(TrainValidationSplitter):
    def __init__(self, icdar_path, validation_percentage):
        super().__init__(validation_percentage)
        self._icdar_path = path.absolute_path(icdar_path)

    def _split_training_data(self, train_dir, val_dir, validation_percentage):
        image_names = path.list_all_images(self._icdar_path)

        total_images = len(image_names)
        nb_train_images = int(total_images * (1 - validation_percentage))

        for i in range(total_images):
            img_name = image_names[i]
            gt_name = _gt_file(img_name)

            link_dir = train_dir if i <= nb_train_images else val_dir

            # Create symbolic links for image and groundtruth in the link directory.
            path.symlink(path.join_path(link_dir, img_name),
                         path.join_path(self._icdar_path, img_name))
            path.symlink(path.join_path(link_dir, gt_name),
                         path.join_path(self._icdar_path, gt_name))


def _gt_file(image_file):
    image_name = path.get_file_name(image_file, with_ext=False)
    return f'gt_{image_name}.txt'
