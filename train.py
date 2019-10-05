from dataset import msra
from east import east, data
from functools import partial


def build_train_model(input_shape=(512, 512, 3)):
    east_model = east.EAST(training=True)
    east_model.build_model(input_shape)
    return east_model


def load_msra(msra_path, batch_size=32, shuffle=True):
    return msra.MSRA(msra_path, batch_size, shuffle)


def process_to_train_data(msra_data,
                          crop_target_size=(512, 512),
                          crop_at_least_one_box_ratio=5/8,
                          random_scales=[0.5, 1.0, 1.5, 2.0],
                          random_angles=[-45, 45]):
    pipeline = [
        partial(data.random_scale, random_scales),
        partial(data.random_rotate, random_angles),
        partial(data.random_crop_with_text_boxes_cropped,
                crop_target_size,
                crop_at_least_one_box_ratio),
        # Ensure that the output image has the cropped size.
        partial(data.pad_image, crop_target_size)
    ]

    msra_iter = iter(msra_data)
    return data.flow_from_generator(msra_iter, pipeline)


if __name__ == "__main__":
    # Load the data.
    msra_data = load_msra(
        '/home/beekill/projects/bookual/datasets/MSRA-TD500/train/', 4)

    # Convert and pre-process images and groundtruth to correct format
    # expected by the model.
    train_generator = process_to_train_data(msra_data)

    # Build the model.
    east_model = build_train_model()
    east_model.summary_model()

    # Begin the training.
    east_model.train(
        train_generator, train_steps_per_epoch=msra_data.steps_per_epoch, epochs=1)
