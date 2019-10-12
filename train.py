import argparse
from dataset import msra
from east import east, preprocessing
from functools import partial
from tensorflow.python.keras.utils.data_utils import OrderedEnqueuer


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--msra-path',
                        dest='msra_path',
                        action='store',
                        required=True,
                        help='Path to MSRA TD500 training dataset.')
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

    return parser.parse_args()


def build_train_model(input_shape=(512, 512, 3)):
    east_model = east.EAST(training=True)
    east_model.build_model(input_shape)
    return east_model


def load_msra(msra_seq, batch_size=32, shuffle=True):
    return msra.MSRASequence(msra_seq, batch_size, shuffle)


def process_to_train_data(msra_seq,
                          crop_target_size=(512, 512),
                          crop_at_least_one_box_ratio=5/8,
                          random_scales=[0.5, 1.0, 1.5, 2.0],
                          random_angles=[-20, 20]):
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


if __name__ == "__main__":
    args = parse_arguments()

    # Load the data.
    msra_seq = load_msra(args.msra_path, args.batch_size)

    # Convert and pre-process images and groundtruth to correct format
    # expected by the model.
    training_seq = process_to_train_data(msra_seq)

    # Build the model.
    east_model = build_train_model()
    east_model.summary_model()

    # Build generator.
    enqueuer = build_training_data_enqueuer(training_seq)
    enqueuer.start()

    # Begin the training.
    data_generator = enqueuer.get()
    east_model.train(data_generator,
                     train_steps_per_epoch=len(msra_seq),
                     epochs=args.epochs)

    # Training finished.
    enqueuer.stop()

    # Save the model.
    east_model.save_model('./east.model')
