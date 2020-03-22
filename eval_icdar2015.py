import argparse
from basenet.resnet50_base import ResNet50Base
from east import east, preprocessing, postprocessing, nms
from dataset import path
from itertools import repeat
import numpy as np
from PIL import Image
import time
import utils


def time_it(name):
    start = time.time()

    def end():
        end = time.time()
        print(f'{name} took {end - start} seconds.')

    return end


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--testdir',
                        required=True,
                        help='Path to the directory contains test images.')
    parser.add_argument('--outdir',
                        required=True,
                        help='Path to the output directory contains output text boxes.')

    parser.add_argument('--checkpoint-path',
                        dest='checkpoint_path',
                        required=True,
                        help='Path to the checkpoint model.')

    parser.add_argument('--max-target-image-size',
                        dest='max_target_image_size',
                        type=int,
                        default=2400,
                        help='Maximum target get size of an image to prevent OOM in gpu.')
    parser.add_argument('--score-threshold',
                        dest='score_threshold',
                        type=float,
                        default=0.5,
                        help='Score threshold to be considered as a valid box.')
    parser.add_argument('--nms-threshold',
                        dest='nms_threshold',
                        type=float,
                        default=0.3,
                        help='Non-maxima suppression threshold between boxes to be considered as the same box.')

    return parser.parse_args()


def build_model(checkpoint_path, image_size):
    model = east.EAST(base_network=ResNet50Base())
    model.build_model(image_size + (3,))
    model.load_model(checkpoint_path)

    return model


def recognize_text(model, images, image_size, score_threshold, nms_threshold):
    def get_text_boxes(predicted):
        predicted_boxes = postprocessing.extract_text_boxes(predicted,
                                                            (512, 512),
                                                            score_threshold)

        predicted_boxes = postprocessing.filter_text_boxes(predicted[:, :, 0],
                                                           predicted_boxes,
                                                           image_size,
                                                           0.5)

        predicted_boxes = [np.append(b[1].flatten(), [b[0]])
                           for b in predicted_boxes]
        return nms.nms_locality(predicted_boxes, nms_threshold)

    time_model = time_it('model')
    predicted = model.predict(np.asarray(images))
    time_model()

    time_nms = time_it('nms')
    boxes = list(map(get_text_boxes, predicted))
    time_nms()

    return boxes


def resize_image_if_neccessary(image, max_target_image_size):
    img_shape = image.size

    if img_shape[0] > max_target_image_size or img_shape[1] > max_target_image_size:
        image, _ = preprocessing.resize_image(max_target_image_size, image, [])

    return image


def test_images_generator(test_dir, max_target_image_size):
    def preprocess_image(image):
        image, _ = preprocessing.square_padding(image, [])
        image = resize_image_if_neccessary(image, max_target_image_size)
        return np.asarray(image)

    def load_image(image_name):
        image = Image.open(path.join_path(test_dir, image_name))
        return image_name, np.asarray(image), preprocess_image(image)

    image_names = path.list_all_images(test_dir)

    for names in utils.chunk(image_names, 1):
        if not names:
            break

        yield list(zip(*map(load_image, names)))


def save_recognition_result(image_name, text_boxes, outdir):
    out_name = f'res_{path.get_file_name(image_name, with_ext=False)}.txt'

    with open(path.join_path(outdir, out_name), 'w', newline='\r\n') as outfile:
        for box in text_boxes:
            b = box[:-1].flatten().astype(int).tolist()
            outfile.write(','.join(str(i) for i in b))
            outfile.write('\n')


def scale_text_boxes(text_boxes, images, orig_images):
    """
    Scale text boxes back to original image size.
    """
    def scale(img_text_boxes, image, orig_image):
        largest_edge = max(orig_image.shape[0], orig_image.shape[1])
        ratio = largest_edge / image.shape[0]

        scaled_boxes = []
        for box in img_text_boxes:
            scaled_boxes.append(np.append(box[:-1] * ratio, box[-1:]))

        return scaled_boxes

    return map(lambda args: scale(*args), zip(text_boxes, images, orig_images))


if __name__ == "__main__":
    args = parse_arguments()

    # Store target image size.
    max_target_image_size = args.max_target_image_size
    assert max_target_image_size % 4 == 0 and max_target_image_size >= 32

    # Create output directory.
    path.make_dirs(args.outdir)

    # Build the model.
    model = build_model(args.checkpoint_path, (None, None))

    # Load and recognize the test images.
    for (names, orig_images, images) in test_images_generator(args.testdir, max_target_image_size):
        print('===')

        # test_images_generator always return 1-image batches.
        w, h, _ = images[0].shape
        text_boxes = recognize_text(model,
                                    images,
                                    (w, h),
                                    args.score_threshold,
                                    args.nms_threshold)

        time_scale_back = time_it('scale_back')
        text_boxes = scale_text_boxes(text_boxes, images, orig_images)
        time_scale_back()

        # Save recognition results to files.
        time_save_results = time_it('save_result')
        for params in zip(names, text_boxes, repeat(args.outdir)):
            save_recognition_result(*params)
        time_save_results()

