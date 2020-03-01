import argparse
from basenet.resnet50_base import ResNet50Base
from basenet.efficientnetb3_base import EfficientNetB3Base
from dataset import msra, icdar
from east import east, preprocessing
from functools import partial
from tensorflow.python.keras.utils.data_utils import OrderedEnqueuer
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
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


def build_training_callbacks(checkpoint_path, tensorboard_path, early_stopping_patience):
    callbacks = []

    if checkpoint_path:
        callbacks.append(
            ModelCheckpoint(checkpoint_path,
                            monitor='val_loss',
                            save_weights_only=True,
                            save_best_only=True)
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
                          patience=early_stopping_patience)
        )

    return callbacks


if __name__ == "__main__":
    args = parse_arguments()

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
        train_seq = process_to_train_data(train_seq)
        val_seq = process_to_val_data(val_seq)

        # Build enqueuers.
        train_enqueuer = build_data_enqueuer(train_seq)
        val_enqueuer = build_data_enqueuer(val_seq)

        # Start enqueuers.
        train_enqueuer.start(workers=args.threads)
        val_enqueuer.start(workers=args.threads)

        # Begin the training.
        try:
            print('\n===== Begin Training =====')
            training_callbacks = build_training_callbacks(args.checkpoint,
                                                          args.tensorboard,
                                                          args.early_stopping_patience)

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
