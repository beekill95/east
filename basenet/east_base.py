import abc


class EastBase(abc.ABC):
    """
    Base network interface for EAST.
    """

    def __init__(self, trainable):
        self._trainable = trainable

    @abc.abstractmethod
    def build(self, input_shape):
        pass

    @abc.abstractmethod
    def input(self):
        """Return the input of the base network."""
        pass

    @abc.abstractmethod
    def stage_1(self):
        """Return the stage 1 of the base network."""
        pass

    @abc.abstractmethod
    def stage_2(self):
        """Return the stage 2 of the base network."""
        pass

    @abc.abstractmethod
    def stage_3(self):
        """Return the stage 3 of the base network."""
        pass

    @abc.abstractmethod
    def stage_4(self):
        """Return the stage 4 of the base network."""
        pass
