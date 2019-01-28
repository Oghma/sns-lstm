"""Module that defines the pooling modules."""
from abc import ABC, abstractmethod
import tensorflow as tf


class Pooling(ABC):
    """Abstract pooling class. Define the interface for the pooling layers."""

    def __init__(self, hparams):
        """Constructor of the Pooling class.

        Args:
          hparams: An HParams instance. hparams must contains gridSize,
            neighborhoodSize, maxNumPed, embeddingSize and rnnSize values.

        """
        self.grid_size = hparams.gridSize
        self.neighborhood_size = hparams.neighborhoodSize
        self.max_num_ped = hparams.maxNumPed

        self.pooling_layer = tf.layers.Dense(
            hparams.embeddingSize,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="Pooling/Layer",
        )

    @abstractmethod
    def pooling(self, pedestrians, coordinates, states, peds_mask, *args):
        """Compute the pooling.

        Args:
          pedestrians: tensor of shape [max_num_ped, 2]. Pedestrians.
          coordinates: tensor of shape [max_num_ped, 2]. Coordinates.
          states: tensor of shape [max_num_ped, rnn_size]. Cell states of the
            LSTM.
          peds_mask: tensor of shape [max_num_ped, max_num_ped]. Grid layer.

        Returns:
          The pooling layer

        """
        pass

    def _get_bounds(self, coordinates):
        """Calculates the bounds of each pedestrian.

        Args:
          coordinates: tensor of shape [max_num_ped, 2]. Coordinates

        Returns:
          tuple containing tensor of shape [max_num_ped, 2] the top left and
            bottom right bounds.

        """
        top_left_x = coordinates[:, 0] - (self.neighborhood_size / 2)
        top_left_y = coordinates[:, 1] + (self.neighborhood_size / 2)
        bottom_right_x = coordinates[:, 0] + (self.neighborhood_size / 2)
        bottom_right_y = coordinates[:, 1] - (self.neighborhood_size / 2)

        top_left = tf.stack([top_left_x, top_left_y], axis=1)
        bottom_right = tf.stack([bottom_right_x, bottom_right_y], axis=1)

        return top_left, bottom_right

    def _grid_pos(self, top_left, coordinates):
        """Calculate the position in the grid layer of the neighbours.

        Args:
          top_left: tensor of shape [max_num_ped * max_num_ped, 2]. Top left
            bound.
          coordinates: tensor of shape [max_num_ped * max_num_ped, 2].
            Coordinates.

        Returns:
          Tensor of shape [max_num_ped * max_num_ped] that is the position in
            the grid layer of the neighbours.

        """
        cell_x = tf.floor(
            ((coordinates[:, 0] - top_left[:, 0]) / self.neighborhood_size)
            * self.grid_size
        )
        cell_y = tf.floor(
            ((top_left[:, 1] - coordinates[:, 1]) / self.neighborhood_size)
            * self.grid_size
        )
        grid_pos = cell_x + cell_y * self.grid_size
        return tf.cast(grid_pos, tf.int32)

    def _repeat(self, tensor):
        """ Repeat each row of the input tensor max_num_ped times

        Args:
          tensor: tensor of shape [max_num_ped, n]
        Returns:
          tensor of shape [max_num_ped * max_num_ped, n]. Repeat each row of the
            input tensor in order to have row1, row1, row1, row2, row2, row2,
            etc

        """
        col_len = tensor.shape[1]
        tensor = tf.expand_dims(tensor, 1)
        # Tensor has now shape [max_num_ped, 1, 2]. Now repeat ech row
        tensor = tf.tile(tensor, (1, self.max_num_ped, 1))
        tensor = tf.reshape(tensor, (-1, col_len))

        return tensor


class SocialPooling(Pooling):
    """Implement the Social layer defined in social LSTM paper"""

    def __init__(self, hparams):
        """Constructor of the Social pooling class.

        Args:
          hparams: An HParams instance. hparams must contains grid_size,
            neighborhood_size, max_num_ped, embedding_size and rnn_size values.

        """
        super().__init__(hparams)
        self.rnn_size = hparams.rnnSize

    def pooling(self, pedestrians, coordinates, states, peds_mask, *args):
        """Compute the social pooling.

        Args:
          pedestrians: tensor of shape [max_num_ped, 2]. Pedestrians.
          coordinates: tensor of shape [max_num_ped, 2]. Coordinates.
          states: tensor of shape [max_num_ped, rnn_size]. Cell states of the
            LSTM.
          peds_mask: tensor of shape [max_num_ped, max_num_ped]. Grid layer.

        Returns:
          The social pooling layer

        """
        top_left, bottom_right = self._get_bounds(pedestrians)

        # Repeat the coordinates in order to have P1, P2, P3, P1, P2, P3
        coordinates = tf.tile(coordinates, (self.max_num_ped, 1))
        # Repeat the hidden states in order to have S1, S2, S3, S1, S2, S3
        states = tf.tile(states, (self.max_num_ped, 1))
        # Repeat the bounds in order to have B1, B1, B1, B2, B2, B2
        top_left = self._repeat(top_left)
        bottom_right = self._repeat(bottom_right)

        grid_layout = self._grid_pos(top_left, coordinates)

        # Find which pedestrians are to include
        x_bound = tf.logical_and(
            (coordinates[:, 0] < bottom_right[:, 0]),
            (coordinates[:, 0] > top_left[:, 0]),
        )
        y_bound = tf.logical_and(
            (coordinates[:, 1] < top_left[:, 1]),
            (coordinates[:, 1] > bottom_right[:, 1]),
        )

        peds_mask = tf.reshape(peds_mask, [self.max_num_ped * self.max_num_ped])
        mask = tf.logical_and(tf.logical_and(x_bound, y_bound), peds_mask)

        # tf.scatter_add works only with 1D tensors. The values in grid_layout
        # are in [0, grid_size * grid_size]. It needs an offset
        total_grid = self.grid_size * self.grid_size
        offset = tf.range(0, total_grid * self.max_num_ped, total_grid)
        offset = tf.reshape(self._repeat(tf.reshape(offset, [-1, 1])), [-1])
        grid_layout = grid_layout + offset

        indices = tf.boolean_mask(grid_layout, mask)

        scattered = tf.reshape(
            tf.scatter_nd(
                tf.expand_dims(indices, 1),
                tf.boolean_mask(states, mask),
                shape=[
                    self.max_num_ped * self.grid_size * self.grid_size,
                    self.rnn_size,
                ],
            ),
            (self.max_num_ped, -1),
        )

        return self.pooling_layer(scattered)


class OccupancyPooling(Pooling):
    """Implement the Occupancy layer defined in social LSTM paper"""

    def __init__(self, hparams):
        """Constructor of the Occupancy pooling class.

        Args:
          hparams: An HParams instance. hparams must contains grid_size,
            neighborhood_size, max_num_ped, embedding_size and rnn_size values.

        """
        super().__init__(hparams)

    def pooling(self, pedestrians, coordinates, states, peds_mask, *args):
        """Compute the occupancy pooling.

        Args:
          pedestrians: tensor of shape [max_num_ped, 2]. Pedestrians.
          coordinates: tensor of shape [max_num_ped, 2]. Coordinates.
          states: tensor of shape [max_num_ped, rnn_size]. Cell states of the
            LSTM.
          peds_mask: tensor of shape [max_num_ped, max_num_ped]. Grid layer.

        Returns:
          The occupancy pooling layer

        """
        top_left, bottom_right = self._get_bounds(pedestrians)

        # Repeat the coordinates in order to have P1, P2, P3, P1, P2, P3
        coordinates = tf.tile(coordinates, (self.max_num_ped, 1))
        # Repeat the bounds in order to have B1, B1, B1, B2, B2, B2
        top_left = self._repeat(top_left)
        bottom_right = self._repeat(bottom_right)

        grid_layout = self._grid_pos(top_left, coordinates)

        # Find which pedestrians are to include
        x_bound = tf.logical_and(
            (coordinates[:, 0] < bottom_right[:, 0]),
            (coordinates[:, 0] > top_left[:, 0]),
        )
        y_bound = tf.logical_and(
            (coordinates[:, 1] < top_left[:, 1]),
            (coordinates[:, 1] > bottom_right[:, 1]),
        )

        peds_mask = tf.reshape(peds_mask, [self.max_num_ped * self.max_num_ped])
        mask = tf.logical_and(tf.logical_and(x_bound, y_bound), peds_mask)

        # tf.scatter_add works only with 1D tensors. The values in grid_layout
        # are in [0, grid_size * grid_size]. It needs an offset
        total_grid = self.grid_size * self.grid_size
        offset = tf.range(0, total_grid * self.max_num_ped, total_grid)
        offset = tf.reshape(self._repeat(tf.reshape(offset, [-1, 1])), [-1])
        grid_layout = grid_layout + offset

        indices = tf.boolean_mask(grid_layout, mask)

        scattered = tf.reshape(
            tf.scatter_nd(
                tf.expand_dims(indices, 1),
                tf.boolean_mask(states, mask),
                shape=[self.max_num_ped * self.grid_size * self.grid_size, 1],
            ),
            (self.max_num_ped, -1),
        )

        return self.pooling_layer(scattered)


class CombinedPooling:
    """Combined pooling class. Define multiple pooling layer combined with each
    other."""

    def __init__(self, hparams):
        """Constructor of the CombinedPooling class.

        Args:
          hparams: An HParams instance. hparams must contains grid_size,
            neighborhood_size, max_num_ped, embedding_size, rnn_size, layers and
            all_peds values.

        """
        self.__layers = []
        self.__all_peds_layers = []

        for layer in hparams.poolingModule:
            if layer == "social":
                social = SocialPooling(hparams)
                if hparams.allPeds:
                    self.__all_peds_layers.append(social)
                else:
                    self.__layers.append(social)
            elif layer == "occupancy":
                self.__all_peds_layers.append(OccupancyPooling(hparams))

    def pooling(
        self, pedestrians, coordinates, states, peds_mask, all_peds, all_peds_mask
    ):
        """Compute the combined pooling.

        Args:
          pedestrians: tensor of shape [max_num_ped, 2]. Pedestrians.
          coordinates: tensor of shape [max_num_ped, 2]. Coordinates.
          states: tensor of shape [max_num_ped, rnn_size]. Cell states of the
            LSTM.
          peds_mask: tensor of shape [max_num_ped, max_num_ped]. Grid layer.
          all_peds: tensor of shape [max_num_ped, embedding_size]. Coordinates
            of all pedestrians.
          all_peds_mask: tensor of shape [max_num_ped, max_num_ped]. Mask of all
            pedestrians.

        Returns:
          The pooling layer

        """
        pooled = []
        for layer in self.__layers:
            pooled.append(layer.pooling(pedestrians, coordinates, states, peds_mask))

        for layer in self.__all_peds_layers:
            pooled.append(layer.pooling(pedestrians, all_peds, states, all_peds_mask))

        return tf.concat(pooled, 1)
