import dataset.utils as data_utils
from east.geometry import rotate_polygon
import numpy as np
import math
# from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
from random import shuffle


def load_msra_td_500(path, batch_size, shuffle=True):
    return iter(MSRA(path, batch_size, shuffle))


class MSRA:
    """
    Load the data in MSRA data set.
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

        # Single thread code.
        images = []
        gts = []
        for i in range(self._step * batch_size, next_index):
            images.append(self._load_img(self._image_files[i]))
            gt = self._load_gt(self._image_files[i])
            box_coordinates = [
                g[:1] + MSRA._convert_geometry_to_coordinates(g[1:]) for g in gt]
            gts.append(box_coordinates)

        self._step += 1

        return images, gts

        # Multi-thread code.
        # def load(url):
        #     img = self._load_img(url)
        #     gt = self._load_gt(url)
        #     box_coordinates = [
        #         g[:1] + MSRA._convert_geometry_to_coordinates(g[1:]) for g in gt]
        #     return img, gt

        # image_names = self._image_files[self._step * batch_size:next_index]
        # thread_pool = ThreadPool(2)
        # results = thread_pool.map(load, image_names)

        # self._step += 1
        # images = [r[0] for r in results]
        # gts = [r[1] for r in results]

        # return [images, gts]

    def _load_gt(self, img_file):
        """
        Load ground truth textboxes into a list.
        Each item in list will be another list:
        [difficulty, x, y, width, height, angle]
        """
        gt_file = self._gt_file(img_file)
        text_boxes = []

        with open(self._full_path(gt_file), 'r') as f:
            for line in f.readlines():
                box_encoded = line.rstrip().split(' ')
                angle = float(box_encoded[-1])
                box = [int(n) for n in box_encoded[1:-1]] + [angle]
                text_boxes.append(box)

        return text_boxes

    def _load_img(self, img_file):
        return np.asarray(Image.open(self._full_path(img_file)))

    def _gt_file(self, img_file):
        img_name = data_utils.get_file_name(img_file, False)
        return f'{img_name}.gt'

    def _full_path(self, file_name):
        return data_utils.join_path(self._data_set_path, file_name)

    @staticmethod
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
