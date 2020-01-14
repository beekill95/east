import argparse
from basenet.resnet50_base import ResNet50Base
from basenet.efficientnetb3_base import EfficientNetB3Base
from dataset import msra, icdar
from east import east, preprocessing
from functools import partial
from tensorflow.python.keras.utils.data_utils import OrderedEnqueuer
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
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
    parser.add_argument('--output',
                        action='store',
                        help='''
                        (Optional) Name of the output model, followed convention of Keras save api.
                        If not present, final model won't be saved.
                        ''')

    return parser.parse_args()


def build_train_model(input_shape=(512, 512, 3)):
    base_network = EfficientNetB3Base()

    east_model = east.EAST(training=True, base_network=base_network)
    east_model.build_model(input_shape)
    return east_model


def load_training_data(args, shuffle=True):
    batch_size = args.batch_size

    if args.msra_path:
        return msra.MSRASequence(args.msra_path, batch_size, shuffle)
    elif args.icdar_2015:
        return icdar.ICDAR2015Sequence(args.icdar_2015, batch_size, shuffle)
    else:
        raise Exception('Neither MSRA nor ICDAR training dataset present.')


def process_to_train_data(msra_seq,
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

    return preprocessing.PreprocessingSequence(msra_seq, pipeline)


def build_training_data_enqueuer(training_seq):
    enqueuer = OrderedEnqueuer(training_seq)
    return enqueuer


def build_training_callbacks(checkpoint_path, tensorboard_path):
    callbacks = []

    if checkpoint_path:
        callbacks.append(
            ModelCheckpoint(checkpoint_path,
                            monitor='mae',
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

    return callbacks


if __name__ == "__main__":
    args = parse_arguments()

    # Check if either checkpoints or output argument present.
    if not args.checkpoint and not args.output:
        warnings.warn('Neither checkpoint or output argument present, train model won\'t be saved!',
                      UserWarning)

    # Load the data.
    data_seq = load_training_data(args, shuffle=True)

    # Convert and pre-process images and groundtruth to correct format
    # expected by the model.
    training_seq = process_to_train_data(data_seq)

    # Build the model.
    east_model = build_train_model()
    east_model.summary_model()

    # Build generator.
    enqueuer = build_training_data_enqueuer(training_seq)
    enqueuer.start(workers=args.threads)

    # Begin the training.
    try:
        print('===== Begin Training =====')

        training_callbacks = build_training_callbacks(args.checkpoint,
                                                      args.tensorboard)

        data_generator = enqueuer.get()
        east_model.train(data_generator,
                         train_steps_per_epoch=len(data_seq),
                         epochs=args.epochs,
                         callbacks=training_callbacks)

        print('===== End Training =====')
    except KeyboardInterrupt:
        print('===== Training Interupted =====')

    # Stop generator.
    enqueuer.stop()

    # Save the model.
    if args.output:
        east_model.save_model(args.output)
