"""Module that defines the pooling modules."""
import tensorflow as tf


class SocialPooling:
    def __init__(
        self,
        grid_size=8,
        neighborhood_size=4,
        max_num_ped=100,
        embedding_size=64,
        rnn_size=128,
    ):
        """Constructor of the SocialPooling class.

        Args:
          grid_size: int or float.
          neighboorhood_size: int or float.
          max_num_ped: int. Maximum number of pedestrian in a single frame.
          embedding_size int. Dimension of the output space of the embedding
            layers.
          rnn_size: int. The number of units in the LSTM cell.
        """
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.max_num_ped = max_num_ped

        with tf.variable_scope("Social_Pooling"):
            self.grid = tf.Variable(
                tf.zeros([max_num_ped * grid_size * grid_size, rnn_size], tf.float32),
                trainable=False,
                name="grid",
            )
            self.pooling_layer = tf.layers.Dense(
                embedding_size,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="Layer",
            )

    def pooling(self, coordinates, states, peds_mask):
        """Compute the social pooling.

        Args:
          coordinates: tensor of shape [max_num_ped, 2]. Coordinates.
          states: tensor of shape [max_num_ped, rnn_size]. Cell states of the
            LSTM.
          peds_mask: tensor of shape [max_num_ped, max_num_ped]. Grid layer.

        Returns:
          The social pooling layer

        """
        top_left, bottom_right = self.__get_bounds(coordinates)

        # Repeat the coordinates in order to have P1, P2, P3, P1, P2, P3
        coordinates = tf.tile(coordinates, (self.max_num_ped, 1))
        # Repeat the hidden states in order to have S1, S2, S3, S1, S2, S3
        states = tf.tile(states, (self.max_num_ped, 1))
        # Repeat the bounds in order to have B1, B1, B1, B2, B2, B2
        top_left = self.__repeat(top_left)
        bottom_right = self.__repeat(bottom_right)

        grid_layout = self.__grid_pos(top_left, coordinates)

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
        offset = tf.reshape(self.__repeat(tf.reshape(offset, [-1, 1])), [-1])
        grid_layout = grid_layout + offset

        with tf.control_dependencies([self.grid.initializer]):
            scattered = tf.reshape(
                tf.scatter_add(
                    self.grid,
                    tf.boolean_mask(grid_layout, mask),
                    tf.boolean_mask(states, mask),
                ),
                (self.max_num_ped, -1),
            )

        return self.pooling_layer(scattered)

    def __get_bounds(self, coordinates):
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

    def __grid_pos(self, top_left, coordinates):
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
        cell_x = (
            tf.floor((coordinates[:, 0] - top_left[:, 0]) / self.neighborhood_size)
            * self.grid_size
        )
        cell_y = (
            tf.floor((top_left[:, 1] - coordinates[:, 1]) / self.neighborhood_size)
            * self.grid_size
        )
        grid_pos = cell_x + cell_y * self.grid_size
        return tf.cast(grid_pos, tf.int32)

    def __repeat(self, tensor):
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