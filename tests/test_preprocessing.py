import east.preprocessing as preprocessing
import numpy as np
from PIL import Image
import random


def test_random_crop_always_have_text_boxes():
    img = Image.new('L', (4000, 3000))

    def generate_random_text_box():
        left = random.randint(0, 3000)
        top = random.randint(0, 2000)

        width = random.randint(0, 999)
        height = random.randint(0, 999)

        return np.array([
            [left, top],
            [left, top + height],
            [left + width, top + height],
            [left + width, top]
        ])

    # Normal cases.
    for i in range(100):
        text_boxes = [
            generate_random_text_box(),
            generate_random_text_box()
        ]

        cropped_img, cropped_text_boxes = preprocessing.random_crop_with_text_boxes_cropped(
            (512, 512), 1.0, img, text_boxes)

        assert len(cropped_text_boxes) > 0
        assert len(cropped_text_boxes) <= 2

    # When the number of boxes exceed 31 boxes.
    for i in range(100):
        text_boxes = [generate_random_text_box() for i in range(35)]

        cropped_img, cropped_text_boxes = preprocessing.random_crop_with_text_boxes_cropped(
            (512, 512), 1.0, img, text_boxes)

        assert len(cropped_text_boxes) > 0
        assert len(cropped_text_boxes) <= 35
