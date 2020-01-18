from basenet.east_base import EastBase
from basenet import utils
import efficientnet.tfkeras as efn
from tensorflow import keras
from tensorflow.keras import backend as K


class EfficientNetB3Base(EastBase):
    def __init__(self, trainable=False):
        super().__init__(trainable)

    def build(self, input_shape):
        self._input = keras.Input(shape=input_shape)
        preprocessed = EfficientNetB3Base._standardize_images(self._input)

        self._model = efn.EfficientNetB3(include_top=False,
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
        return utils.get_output(self._model, 'block3a_expand_activation')

    def stage_2(self):
        self._assert_model_built()
        return utils.get_output(self._model, 'block4a_expand_activation')

    def stage_3(self):
        self._assert_model_built()
        return utils.get_output(self._model, 'block6a_expand_activation')

    def stage_4(self):
        self._assert_model_built()
        return utils.get_output(self._model, 'top_activation')

    def _assert_model_built(self):
        assert self._model, 'EfficientNet B3 is not constructed.'

    @staticmethod
    def _standardize_images(images, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        channels = []

        for i in range(3):
            img = (images[:, :, :, i] / 255. - means[i]) / stds[i]
            channels.append(K.expand_dims(img, axis=-1))

        return keras.layers.concatenate(channels, axis=-1)
