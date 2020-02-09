import argparse
from basenet.resnet50_base import ResNet50Base
from basenet.efficientnetb3_base import EfficientNetB3Base
from dataset import msra
from east import east, postprocessing, nms
import numpy as np
from PIL import Image, ImageDraw


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image',
                        required=True,
                        help='Path to image to recognize.')
    parser.add_argument('--output',
                        required=True,
                        help='Path to the output image.')

    parser.add_argument('--checkpoint-path',
                        dest='checkpoint_path',
                        required=True,
                        help='Path to the checkpoint model.')

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
    # base = EfficientNetB3Base()
    base = ResNet50Base()

    model = east.EAST(base_network=base)
    model.build_model(image_size + (3,))
    model.load_model(checkpoint_path)

    return model


def recognize_text(model, images, image_size, score_threshold, nms_threshold):
    predicted = model.predict(np.asarray(images))

    predicted_boxes = postprocessing.extract_text_boxes(predicted[0],
                                                        (512, 512),
                                                        score_threshold)

    predicted_boxes = postprocessing.filter_text_boxes(predicted[0][:, :, 0],
                                                       predicted_boxes,
                                                       image_size,
                                                       0.5)

    predicted_boxes = [np.append(b[1].flatten(), [b[0]])
                       for b in predicted_boxes]
    return nms.nms_locality(predicted_boxes, nms_threshold)


def square_padding(image):
    """
    Pad image to have square image.
    """
    w, h = image.size

    if w == h:
        return image

    padded_img = Image.new(image.mode, (w, w) if w > h else (h, h))
    padded_img.paste(image)

    return padded_img


if __name__ == "__main__":
    args = parse_arguments()
    orig_img = Image.open(args.image)
    img = square_padding(orig_img)

    img_target_size = (1024, 1024)
    img = img.resize(img_target_size)
    img = np.asarray(img)

    model = build_model(args.checkpoint_path, img_target_size)
    text_boxes = recognize_text(model,
                                [img],
                                img_target_size,
                                args.score_threshold,
                                args.nms_threshold)

    # Draw predicted image.
    # predict = Image.fromarray(img)
    predict = orig_img
    pred_draw = ImageDraw.Draw(orig_img)
    for b in text_boxes:
        r = b[:-1].reshape(4, 2)
        abc = orig_img.size[1] if orig_img.size[1] > orig_img.size[0] else orig_img.size[0]
        r[:, 0] *= abc / 1024
        r[:, 1] *= abc / 1024
        r = r.flatten().astype(int).tolist()
        pred_draw.polygon(r)

    predict.save(args.output)
