from basenet import east_base, utils
from tensorflow import keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50


class ResNet50Base(east_base.EastBase):
    """
    Resnet 50 base network for EAST.
    """

    def __init__(self, trainable=False):
        super().__init__(trainable)

    def build(self, input_shape):
        self._input = keras.Input(shape=input_shape)
        preprocessed = ResNet50Base._mean_pixel_subtraction(self._input)

        self._model = ResNet50(include_top=False,
                               weights='imagenet',
                               input_tensor=preprocessed)

        if not self._trainable:
            self._model.trainable = False

    def summary(self):
        self._assert_model_built()
        self._model.summary()

    def input(self):
        self._assert_model_built()
        return self._input

    def stage_1(self):
        self._assert_model_built()
        return utils.get_output(self._model, 'conv2_block3_out')

    def stage_2(self):
        self._assert_model_built()
        return utils.get_output(self._model, 'conv3_block4_out')

    def stage_3(self):
        self._assert_model_built()
        return utils.get_output(self._model, 'conv4_block6_out')

    def stage_4(self):
        self._assert_model_built()
        return utils.get_output(self._model, 'conv5_block3_out')

    def _assert_model_built(self):
        assert self._model, 'Base network is not constructed.'

    @staticmethod
    def _mean_pixel_subtraction(images, means=[123.68, 116.78, 103.94]):
        channels = []
        for i in range(3):
            img = images[:, :, :, i] - means[i]
            channels.append(K.expand_dims(img, axis=-1))

        # Concatenate these layers, also switch from RGB to BGR,
        # similar to Keras's preprocess_input function.
        return keras.layers.concatenate(channels[::-1], axis=-1)
