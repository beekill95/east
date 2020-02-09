import argparse
from basenet.resnet50_base import ResNet50Base
from east import east, preprocessing, postprocessing, nms
from dataset import path
from itertools import repeat
import numpy as np
from PIL import Image
import utils


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
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        type=int,
                        default=32,
                        help='Number of images per batch.')

    parser.add_argument('--target-image-size',
                        dest='target_image_size',
                        type=int,
                        default=1024,
                        help='Destination test image size. Must be multiple of 4 and larger than 32.')
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

    predicted = model.predict(np.asarray(images))
    return list(map(get_text_boxes, predicted))


def test_images_generator(test_dir, batch_size, target_image_size):
    def preprocess_image(image):
        image, _ = preprocessing.square_padding(image, [])
        image, _ = preprocessing.resize_image(target_image_size, image, [])
        return np.asarray(image)

    def load_image(image_name):
        image = Image.open(path.join_path(test_dir, image_name))
        return image_name, np.asarray(image), preprocess_image(image)

    image_names = path.list_all_images(test_dir)

    for names in utils.chunk(image_names, batch_size):
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
    target_image_size = args.target_image_size
    assert target_image_size % 4 == 0 and target_image_size >= 32
    target_image_size = (target_image_size, ) * 2

    # Create output directory.
    path.make_dirs(args.outdir)

    # Build the model.
    model = build_model(args.checkpoint_path, target_image_size)

    # Load and recognize the test images.
    for (names, orig_images, images) in test_images_generator(args.testdir, args.batch_size, target_image_size):
        text_boxes = recognize_text(model,
                                    images,
                                    target_image_size,
                                    args.score_threshold,
                                    args.nms_threshold)

        text_boxes = scale_text_boxes(text_boxes, images, orig_images)

        # Save recognition results to files.
        for params in zip(names, text_boxes, repeat(args.outdir)):
            save_recognition_result(*params)
