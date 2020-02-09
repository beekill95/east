import abc
from dataset import path
import os
from tempfile import TemporaryDirectory


class TrainValidationSplitter(abc.ABC):
    def __init__(self, validation_percentage=0.2):
        assert 0 <= validation_percentage < 1, 'Validation percentage should be in between 0 and 1'
        self._validation_percentage = validation_percentage

    def __enter__(self):
        self._tempdir = TemporaryDirectory()

        train_dir = self._create_dir('train')
        val_dir = self._create_dir('val')
        self._split_training_data(train_dir,
                                  val_dir,
                                  self._validation_percentage)

        return train_dir, val_dir

    def __exit__(self, type, value, tb):
        self._tempdir.cleanup()

    @abc.abstractmethod
    def _split_training_data(self, train_dir, val_dir, validation_percentage):
        """
        Subclass should inherit this method to split the data into two directories (train and validation),
        and return the path of those directories.

        :param train_dir: the directory that will contain the train data.
        :param val_dir: the directory that will contain the validaton data.
        :param validation_percentage: the percentage (0 - 1) of the validation set.
        :return: paths to the two directories train and validation.
        """
        raise Exception('This method is not implemented.')

    def _create_dir(self, name):
        d = path.join_path(self._tempdir.name, name, '')
        path.make_dirs(d)

        return d
