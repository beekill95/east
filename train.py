import argparse
from basenet.resnet50_base import ResNet50Base
from basenet.efficientnetb3_base import EfficientNetB3Base
from dataset import msra, icdar
from east import east, preprocessing, postprocessing
from functools import partial
import numpy as np
from PIL import Image, ImageDraw
import random
from tensorflow.python.keras.utils.data_utils import OrderedEnqueuer
from tensorflow.python.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
import warnings


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--msra-path',
                        dest='msra_path',
                        action='store',
                        help='Path to MSRA TD500 training dataset.')
    parser.add_argument('--icdar-2015',
                        dest='icdar_2015',
                        action='store',
                        help='Path to ICDAR 2015 training dataset.')
    parser.add_argument('--validation-percentage',
                        dest='validation_percentage',
                        action='store',
                        type=float,
                        default=0.2,
                        help='Percentage to split the data into training and validation.')

    parser.add_argument('--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=32,
                        help='Image batch size.')
    parser.add_argument('--epochs',
                        action='store',
                        type=int,
                        default=100,
                        help='Number or training epochs.')
    parser.add_argument('--threads',
                        action='store',
                        type=int,
                        default=2,
                        help='Number of threads to run when doing preprocessing')
    parser.add_argument('--checkpoint',
                        action='store',
                        help='''
                        (Optional) Path to the directory to store the checkpoints as well as naming of the checkpoint.
                        For more information, please check Keras API. If not present, checkpoints won't be saved.
                        ''')
    parser.add_argument('--tensorboard',
                        action='store',
                        help='''
                        (Optional) Path to directory to store the tensorboard.
                        If not present, tensorboard won't be saved.
                        ''')
    parser.add_argument('--early-stopping-patience',
                        action='store',
                        dest='early_stopping_patience',
                        type=int,
                        help='''
                        (Optional) Number of epochs to wait before ending the training session when loss ceases to decrease.
                        If not present, this won't be applied.
                        ''')
    parser.add_argument('--output',
                        action='store',
                        help='''
                        (Optional) Name of the output model, followed convention of Keras save api.
                        If not present, final model won't be saved.
                        ''')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='Use Weight & Bias to track experiments.')
    parser.add_argument('--random-seed',
                        action='store',
                        dest='random_seed',
                        type=int,
                        default=77,
                        help='Set random seed for for reproducible training.')

    return parser.parse_args()


def build_train_model(input_shape=(512, 512, 3)):
    base_network = ResNet50Base()

    east_model = east.EAST(training=True, base_network=base_network)
    east_model.build_model(input_shape)
    return east_model


def load_data_sequences(args, train_dir, val_dir, shuffle=True):
    batch_size = args.batch_size

    if args.msra_path:
        train_seq = msra.MSRASequence(train_dir, batch_size, shuffle)
        val_seq = msra.MSRASequence(val_dir, batch_size, False)
    elif args.icdar_2015:
        train_seq = icdar.ICDAR2015Sequence(train_dir, batch_size, shuffle)
        val_seq = icdar.ICDAR2015Sequence(val_dir, batch_size, False)
    else:
        raise Exception('Neither MSRA nor ICDAR training dataset present.')

    return train_seq, val_seq


def training_data_splitter(args):
    if args.msra_path:
        return msra.MSRATrainValidationSplitter(args.msra_path, args.validation_percentage)
    elif args.icdar_2015:
        return icdar.ICDAR2015TrainValidationSplitter(args.icdar_2015, args.validation_percentage)
    else:
        raise Exception('Neither MSRA nor ICDAR training dataset present.')


def process_to_train_data(train_seq,
                          crop_target_size=(512, 512),
                          crop_at_least_one_box_ratio=5/8,
                          random_scales=[0.5, 0.75, 1.0, 1.25],
                          random_angles=[-5, 5]):
    pipeline = [
        partial(preprocessing.random_scale, random_scales),
        partial(preprocessing.random_rotate, random_angles),
        partial(preprocessing.random_crop_with_text_boxes_cropped,
                crop_target_size,
                crop_at_least_one_box_ratio),
        # Ensure that the output image has the cropped size.
        partial(preprocessing.pad_image, crop_target_size)
    ]

    return preprocessing.PreprocessingSequence(train_seq, pipeline)


def process_to_val_data(val_seq, target_size=(512, 512)):
    pipeline = [
        partial(preprocessing.square_padding),
        partial(preprocessing.resize_image, target_size=target_size)
    ]

    return preprocessing.PreprocessingSequence(val_seq, pipeline)


def build_data_enqueuer(data_seq):
    enqueuer = OrderedEnqueuer(data_seq)
    return enqueuer


class WandbImageLogger(Callback):
    def __init__(self,
                 preprocessed_train_images,
                 train_image_gts,
                 preprocessed_test_images=[],
                 test_image_gts=[]):
        super(WandbImageLogger, self).__init__()

        assert len(preprocessed_train_images) < 5 and len(
            preprocessed_test_images) < 5

        self.train_images = np.asarray(preprocessed_train_images)
        self.train_image_gts = np.asarray(train_image_gts)
        self.test_images = np.asarray(preprocessed_test_images)
        self.test_image_gts = np.asarray(test_image_gts)

    def on_epoch_end(self, epoch, logs=None):
        # Here, we will run prediction on train images and test images.
        nb_train_images = len(self.train_images)
        prediction = self.model.predict(np.concatenate((self.train_images,
                                                        self.test_images)))

        images = []
        score_maps = []
        ious = []
        angles = []

        for i in range(len(prediction)):
            is_train_image = i < nb_train_images
            image_type = 'train' if is_train_image else 'test'

            pred = prediction[i]
            image = (self.train_images[i]
                     if is_train_image
                     else self.test_images[i - nb_train_images])
            gt = (self.train_image_gts[i]
                  if is_train_image
                  else self.test_image_gts[i - nb_train_images])

            images.append(self._box_predictions(image, pred, i, image_type))
            score_maps.append(self._score_map(pred, i, image_type))
            ious.append(self._iou(pred, gt, i, image_type))
            angles.append(self._angle(pred, gt, i, image_type))

        wandb.log({'predictions': images}, step=epoch, commit=False)
        wandb.log({'score maps': score_maps}, step=epoch, commit=False)
        wandb.log({'ious': ious}, step=epoch, commit=False)
        wandb.log({'angles': angles}, step=epoch, commit=False)

    def _box_predictions(self, image, prediction, ith, image_type):
        boxes = postprocessing.extract_text_boxes(prediction,
                                                  (512, 512),
                                                  0.5)

        # Randomly choose at maximum 10 boxes to draw.
        nb_boxes = min(len(boxes), 10)
        box_indices = np.random.permutation(len(boxes))[:nb_boxes]

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        for nth, box_idx in enumerate(box_indices):
            box = boxes[box_idx][1].flatten().astype(np.int).tolist()
            draw.polygon(box)

        return wandb.Image(image, caption=f'{image_type}_image_{ith}')

    def _score_map(self, prediction, ith, image_type):
        img = Image.fromarray(prediction[:, :, 0] * 255).convert('L')
        return wandb.Image(img, caption=f'{image_type}_score_map_{ith}')

    def _iou(self, prediction, gt, ith, image_type):
        def area(geometry):
            width = geometry[:, :, 0] + geometry[:, :, 2]
            height = geometry[:, :, 1] + geometry[:, :, 3]
            return width * height

        def intersection(pred_geometry, gt_geometry):
            intersect = np.minimum(pred_geometry, gt_geometry)
            return area(intersect)

        # We only care about pixels inside the boxes.
        true_mask = gt[:, :, 0]

        pred_geometry = prediction[:, :, 1:5]
        gt_geometry = gt[:, :, 1:5]

        intersect_area = intersection(pred_geometry, gt_geometry)
        union_area = area(pred_geometry) + area(gt_geometry) - intersect_area

        iou = intersect_area / union_area
        iou = iou * true_mask

        img = Image.fromarray(iou * 255).convert('L')
        return wandb.Image(img, caption=f'{image_type}_iou_{ith}')

    def _angle(self, prediction, gt, ith, image_type):
        # We only care about pixels inside the boxes.
        true_mask = gt[:, :, 0]

        gt_angle = gt[:, :, -1]
        pred_angle = prediction[:, :, -1]

        angle_diff = np.cos(pred_angle - gt_angle)
        angle_diff = angle_diff * true_mask

        img = Image.fromarray(angle_diff * 255).convert('L')
        return wandb.Image(img, caption=f'{image_type}_angle_{ith}')


def build_training_callbacks(checkpoint_path, tensorboard_path, early_stopping_patience, wandb):
    callbacks = []

    if checkpoint_path:
        callbacks.append(
            ModelCheckpoint(checkpoint_path,
                            monitor='val_loss',
                            save_weights_only=True,
                            save_best_only=True,
                            verbose=1)
        )

    if tensorboard_path:
        callbacks.append(
            TensorBoard(log_dir=tensorboard_path,
                        write_graph=True,
                        write_images=True,
                        update_freq='epoch')
        )

    if early_stopping_patience:
        callbacks.append(
            EarlyStopping(monitor='val_loss',
                          patience=early_stopping_patience,
                          verbose=1)
        )

    if wandb:
        from wandb.keras import WandbCallback
        callbacks.append(WandbCallback())

    return callbacks


if __name__ == "__main__":
    args = parse_arguments()

    # Set random seed.
    if args.random_seed:
        random.seed(args.random_seed)

    # Only import wandb when required.
    if args.wandb:
        import wandb
        wandb.init(project='my_east')
        wandb.config.update(args)

    # Check if either checkpoints or output argument present.
    if not args.checkpoint and not args.output:
        warnings.warn('Neither checkpoint or output argument present, train model won\'t be saved!',
                      UserWarning)

    # Build the model.
    east_model = build_train_model()
    east_model.summary_model()

    # Split the data into train and validation set.
    with training_data_splitter(args) as (train_dir, val_dir):
        # Load the data.
        train_seq, val_seq = load_data_sequences(args, train_dir, val_dir)

        # Convert and pre-process images and groundtruth to correct format
        # expected by the model.
        # train_seq = process_to_train_data(train_seq)
        train_seq = process_to_val_data(train_seq)
        val_seq = process_to_val_data(val_seq)

        training_callbacks = []
        if args.wandb:
            # Get some sample images.
            sample_train_images = train_seq[0][0][:5]
            sample_train_image_gts = train_seq[0][1][:5]
            sample_val_images = val_seq[0][0][:5]
            sample_val_image_gts = val_seq[0][1][:5]

            training_callbacks = [WandbImageLogger(sample_train_images,
                                                   sample_train_image_gts,
                                                   sample_val_images,
                                                   sample_val_image_gts)]

        # Build enqueuers.
        train_enqueuer = build_data_enqueuer(train_seq)
        val_enqueuer = build_data_enqueuer(val_seq)

        # Start enqueuers.
        train_enqueuer.start(workers=args.threads)
        val_enqueuer.start(workers=args.threads)

        # Begin the training.
        try:
            print('\n===== Begin Training =====')
            training_callbacks += build_training_callbacks(args.checkpoint,
                                                           args.tensorboard,
                                                           args.early_stopping_patience,
                                                           args.wandb)

            east_model.train(train_generator=train_enqueuer.get(),
                             train_steps_per_epoch=len(train_seq),
                             epochs=args.epochs,
                             callbacks=training_callbacks,
                             validation_generator=val_enqueuer.get(),
                             validation_steps_per_epoch=len(val_seq))

            print('\n===== End Training =====')
        except KeyboardInterrupt:
            print('\n===== Training Interupted =====')

        # Stop enqueuers.
        train_enqueuer.stop()
        val_enqueuer.stop()

    # Save the model.
    if args.output:
        east_model.save_model(args.output)
