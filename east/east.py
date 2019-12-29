from east.loss import rbox_geometry_loss, score_map_loss
from east.rbox import decode_rbox
from math import pi
from tensorflow import keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50


class EAST:
    def __init__(self, training=False):
        self._training = training

    def build_model(self, input_shape, output_geometry='RBOX'):
        """
        Build east model.

        :param input_shape: a tuple shows shape of input images.
        :param output_geometry: geometry use for output layers. Available geometries are RBOX and QUAD.
        """
        self._build_base_network(input_shape)
        self._build_feature_merging_blocks()
        output = self._build_output_layers(output_geometry)

        self._east_model = keras.Model(self._input, output)

        if self._training:
            self._east_model.compile(optimizer='adam',
                                     loss=self._total_loss(),
<< << << < HEAD
                                     metrics=[keras.metrics.mae, 'accuracy'])


== == == =
                                     metrics = [keras.metrics.mae])
>> >>>> > Multiply geometric score with ground truth to limit loss to valid text area.

    def load_model(self, weight_path):
        self._assert_model_initialized()
        self._east_model.load_weights(weight_path)

    def save_model(self, weight_path):
        self._assert_model_initialized()
        self._east_model.save_weights(weight_path)

    def summary_model(self):
        self._assert_model_initialized()
        self._east_model.summary()

    def train(self,
              train_generator,
              train_steps_per_epoch,
              epochs=100,
              verbosity=1,
              callbacks=None,
              validation_generator=None,
              validation_steps_per_epoch=None):
        """
        Train the EAST model.
        All the parameters are closely matched those of Keras API.
        """
        self._assert_model_initialized()
        assert self._training, "EAST model is not initialized to be trained"

        self._east_model.fit_generator(train_generator,
                                       train_steps_per_epoch,
                                       epochs,
                                       verbosity,
                                       callbacks,
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps_per_epoch)

    def predict(self, image, score_map_threshold=0.5):
        self._assert_model_initialized()

    def _build_base_network(self, input_shape):
        self._input = keras.Input(shape=input_shape)
        preprocessed = EAST._mean_pixel_subtraction(self._input)
        self._base_network = ResNet50(include_top=False,
                                      weights='imagenet',
                                      input_tensor=preprocessed)

        # Freeze base network weights.
        self._base_network.trainable = False

        self._stage_1 = get_output_tensor(self._base_network,
                                          'conv2_block3_out')
        self._stage_2 = get_output_tensor(self._base_network,
                                          'conv3_block4_out')
        self._stage_3 = get_output_tensor(self._base_network,
                                          'conv4_block6_out')
        self._stage_4 = get_output_tensor(self._base_network,
                                          'conv5_block3_out')

    def _build_feature_merging_blocks(self):
        self._block_1 = EAST._feature_merging_block(self._stage_4,
                                                    self._stage_3, 128)
        self._block_2 = EAST._feature_merging_block(self._block_1,
                                                    self._stage_2, 64)
        self._block_3 = EAST._feature_merging_block(self._block_2,
                                                    self._stage_1, 32)
        self._block_4 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(self._block_3)

    def _build_output_layers(self, output_geometry):
        self._output_geometry = output_geometry
        feature = self._block_4

        # Scores.
        self._scores = keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation='sigmoid',
            name='scores'
        )(feature)

        if output_geometry == 'RBOX':
            # 4 filters for aabb coordinates, last filter for angle.
            geometry = keras.layers.Conv2D(
                filters=4,
                kernel_size=(1, 1),
                activation='sigmoid',
                name='rbox_geometry'
            )(feature)

            angle = keras.layers.Conv2D(
                filters=1,
                kernel_size=(1, 1),
                activation='sigmoid',
                name='rbox_angle'
            )(feature)
            angle = (angle - 0.5) * 2 * pi

            self._rbox_geometry = keras.layers.concatenate([geometry, angle])
            return keras.layers.concatenate([self._scores, self._rbox_geometry])
        elif output_geometry == 'QUAD':
            self._quad_coords = keras.layers.Conv2D(
                filters=8,
                kernel_size=(1, 1),
                activation='sigmoid',
                name='quad_coords'
            )(feature)

            return keras.layers.concatenate([self._scores, self._quad_coords])

        raise ValueError(f'Unknown geometry type {output_geometry} for EAST detector.'
                         'Available geometries are RBOX and QUAD')

    def _total_loss(self, geometry_lambda=1):
        assert self._output_geometry == 'RBOX', f'{self._output_geometry} loss is not implemented'

        def mean(loss_tensor):
            return K.mean(loss_tensor, axis=[1, 2])

        def loss(y_true, y_pred):
            true_mask = y_true[:, :, :, 0]
            score_loss = score_map_loss(true_mask, y_pred[:, :, :, 0])

            if self._output_geometry == 'RBOX':
                gt_rbox_geometry = y_true[:, :, :, 1:]
                pred_rbox_geometry = y_pred[:, :, :, 1:]

                geometry_loss = rbox_geometry_loss(gt_rbox_geometry,
                                                   pred_rbox_geometry)

            return mean(score_loss) + geometry_lambda * mean(true_mask * geometry_loss)

        return loss

    def _assert_model_initialized(self):
        assert self._east_model is not None, "EAST model is not initialized"

    @staticmethod
    def _feature_merging_block(input_tensor, concat_tensor, filters):
        unpooled_tensor = unpool_layer(input_tensor)
        concated_tensor = concat_layer(unpooled_tensor, concat_tensor)

        conv_1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            activation='relu',
            padding='same'
        )(concated_tensor)

        conv_3 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        )(conv_1)

        return conv_3

    @staticmethod
    def _mean_pixel_subtraction(images, means=[123.68, 116.78, 103.94]):
        channels = []
        for i in range(3):
            img = images[:, :, :, i] - means[i]
            channels.append(K.expand_dims(img, axis=-1))

        # Concatenate these layers, also switch from RGB to BGR,
        # similar to Keras's preprocess_input function.
        return keras.layers.concatenate(channels[::-1], axis=-1)


def unpool_layer(input_tensor):
    return keras.layers.UpSampling2D()(input_tensor)


def concat_layer(a_tensor, b_tensor):
    return keras.layers.concatenate([a_tensor, b_tensor])


def get_output_tensor(model, name):
    return model.get_layer(name).output
