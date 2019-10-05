from tensorflow import keras as keras
from tensorflow.python.keras import backend as K
from east.loss import rbox_geometry_loss, score_map_loss
from east.rbox import decode_rbox


# TODO: allow save checkpoints after each n epochs
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

        self._east_model = keras.Model(self._base_network.input, output)

        if self._training:
            self._east_model.compile(
                optimizer='adam',
                loss=self._total_loss(),
                metrics=['accuracy'])

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

        self._east_model.fit_generator(
            train_generator,
            train_steps_per_epoch,
            epochs,
            verbosity,
            callbacks,
            validation_data=validation_generator,
            validation_steps=validation_steps_per_epoch)

    def predict(self):
        self._assert_model_initialized()
        pass

    def _build_base_network(self, input_shape):
        self._base_network = keras.applications.ResNet50(
            include_top=False, weights='imagenet', input_shape=input_shape)

        # Freeze base network weights.
        self._base_network.trainable = False

        self._stage_1 = get_output_tensor(self._base_network, 'activation_9')
        self._stage_2 = get_output_tensor(self._base_network, 'activation_21')
        self._stage_3 = get_output_tensor(self._base_network, 'activation_39')
        self._stage_4 = get_output_tensor(self._base_network, 'activation_48')

    def _build_feature_merging_blocks(self):
        self._block_1 = EAST._feature_merging_block(
            self._stage_4, self._stage_3, 128)
        self._block_2 = EAST._feature_merging_block(
            self._block_1, self._stage_2, 64)
        self._block_3 = EAST._feature_merging_block(
            self._block_2, self._stage_1, 32)
        self._block_4 = keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu', padding='same')(self._block_3)

    def _build_output_layers(self, output_geometry):
        self._output_geometry = output_geometry
        feature = self._block_4

        # Scores.
        self._scores = keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), activation='relu', name='scores')(feature)

        if output_geometry == 'RBOX':
            # 4 filters for aabb coordinates, last filter for angle.
            self._rbox_geometry = keras.layers.Conv2D(
                filters=5, kernel_size=(1, 1), activation='relu', name='rbox_coords')(feature)

            return keras.layers.concatenate([self._scores, self._rbox_geometry])
        elif output_geometry == 'QUAD':
            self._quad_geometry = keras.layers.Conv2D(
                filters=8, kernel_size=(1, 1), activation='relu', name='quad_coords')(feature)

            return keras.layers.concatenate([self._scores, self._quad_coords])

        raise ValueError(
            f'Unknown geometry type {output_geometry} for EAST detector.'
            'Available geometries are RBOX and QUAD')

    def _total_loss(self, geometry_lambda=1):
        assert self._output_geometry == 'RBOX', f'{self._output_geometry} loss is not implemented'

        def loss(y_true, y_pred):
            score_loss = score_map_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0])

            if self._output_geometry == 'RBOX':
                gt_rbox_geometry = y_true[:, :, :, 1:]
                pred_rbox_geometry = y_pred[:, :, :, 1:]

                geometry_loss = rbox_geometry_loss(
                    gt_rbox_geometry, pred_rbox_geometry)

            return K.mean(score_loss + geometry_lambda * geometry_loss, axis=[1, 2])

        return loss

    def _assert_model_initialized(self):
        assert self._east_model is not None, "EAST model is not initialized"

    @staticmethod
    def _feature_merging_block(input_tensor, concat_tensor, filters):
        unpooled_tensor = unpool_layer(input_tensor)
        concated_tensor = concat_layer(unpooled_tensor, concat_tensor)
        conv_1 = keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), activation='relu', padding='same')(concated_tensor)
        conv_3 = keras.layers.Conv2D(
            filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(conv_1)

        return conv_3


def unpool_layer(input_tensor):
    return keras.layers.UpSampling2D()(input_tensor)


def concat_layer(a_tensor, b_tensor):
    return keras.layers.concatenate([a_tensor, b_tensor])


def get_output_tensor(model, name):
    return model.get_layer(name).output
