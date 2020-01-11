import dataset.utils as data_utils
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
        self._image_paths = data_utils.list_all_images(self._icdar_path)

    def __len__(self):
        return int(len(self._image_paths) / self._batch_size)

    def __getitem__(self, index):
        batch_size = self._batch_size
        max_steps = len(self)

        next_index = ((index + 1) * batch_size
                      if index + 1 < max_steps
                      else len(self._image_paths))

        img_files = self._image_paths[index * batch_size:next_index]

        def gt_file(image_file):
            image_name = data_utils.get_file_name(image_file, with_ext=False)
            return f'gt_{image_name}.txt'

        def load_gt(gt_path):
            text_boxes = []
            with open(gt_path, 'r', encoding='utf-8-sig') as file:
                for line in file:
                    # Get the first 8 box coordinates and convert to location.
                    coords = line.split(',')[:8]
                    coords = [int(c) for c in coords]

                    # Append 0 for dummy difficulty.
                    text_boxes.append([0, *coords])

            return text_boxes

        def load_image(img_path):
            return np.asarray(Image.open(img_path))

        def load_image_gt(image_file):
            img_path = data_utils.join_path(self._icdar_path, image_file)
            gt_path = data_utils.join_path(self._icdar_path,
                                           gt_file(image_file))

            return load_image(img_path), load_gt(gt_path)

        results = (load_image_gt(f) for f in img_files)
        return list(zip(*results))

    def on_epoch_end(self):
        if self._shuffle:
            shuffle(self._image_paths)
