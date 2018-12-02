"""Module that defines the functions for the position estimation"""
import math
import tensorflow as tf


def social_train_position_estimate(cell_output, coordinates_gt, output_size, *args):
    """Calculate the probability density function in training phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Ground truth
        coordinates.
      output_size: int. Dimension of the output size.

    Returns:
      tuple containing a tensor of shape [max_num_ped, 2] that contains the pdf.

    """
    # Calculate the new coordinates based on Graves (2013) equations. Assume a
    # bivariate Gaussian distribution.
    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 - 22
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(std_x)
        std_y = tf.exp(std_y)
        rho = tf.tanh(rho)

        # Equations 24 & 25
        stds = tf.multiply(std_x, std_y)
        rho_neg = tf.subtract(1.0, tf.square(rho))

        # Calculate Z
        z_num1 = tf.subtract(coordinates_gt[:, 0], mu_x)
        z_num2 = tf.subtract(coordinates_gt[:, 1], mu_y)
        z_num3 = tf.multiply(2.0, tf.multiply(rho, tf.multiply(z_num1, z_num2)))
        z = (
            tf.square(tf.div(z_num1, std_x))
            + tf.square(tf.div(z_num2, std_y))
            - tf.div(z_num3, stds)
        )

        # Calculate N
        n_num = tf.exp(tf.div(-z, 2 * rho_neg))
        n_den = tf.multiply(
            2.0, tf.multiply(math.pi, tf.multiply(stds, tf.sqrt(rho_neg)))
        )
        return tf.div(n_num, n_den), None


def social_train_position_estimate_stabilized(
    cell_output, coordinates_gt, output_size, *args
):
    """Calculate the probability density function in training phase. Stabilized
    version.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Ground truth
        coordinates.
      output_size: int. Dimension of the output size.

    Returns:
      tuple containing a tensor of shape [max_num_ped, 2] that contains the pdf.

    """
    # Calculate the new coordinates based on Graves (2013) equations. Assume a
    # bivariate Gaussian distribution.
    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 -> 22
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(tf.abs(std_x))
        std_y = tf.exp(tf.abs(std_y))
        rho = tf.tanh(rho) * 0.99

        # Equations 24 & 25
        stds = tf.math.softplus(tf.multiply(std_x, std_y))
        rho_neg = tf.subtract(1.0, tf.square(rho))

        var_x = tf.math.softplus(tf.square(std_x))
        var_y = tf.math.softplus(tf.square(std_y))

        # Calculate Z
        z_num1 = tf.log(1 + tf.abs(tf.subtract(coordinates_gt[:, 0], mu_x)))
        z_num2 = tf.log(1 + tf.abs(tf.subtract(coordinates_gt[:, 1], mu_y)))
        z_num3 = tf.multiply(2.0, tf.multiply(rho, tf.multiply(z_num1, z_num2)))
        z = (
            tf.div(tf.square(z_num1), var_x)
            + tf.div(tf.square(z_num2), var_y)
            - tf.div(z_num3, stds)
        )

        # Calculate N
        epsilon = 1e-20
        n_num = tf.exp(tf.div(-z, 2 * rho_neg))
        n_den = (
            tf.multiply(2.0, tf.multiply(math.pi, tf.multiply(stds, tf.sqrt(rho_neg))))
            + epsilon
        )
        return tf.div(n_num, n_den), None


def social_sample_position_estimate(
    cell_output, coordinates_gt, output_size, layer_output
):
    """Calculate the coordinates in sampling phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Ground truth
        coordinates.
      output_size: int. Dimension of the output size.
      layer_output: tf.layer instance. Layer used for process the new
        coordinates sampled.

    Returns:
      tuple containing two tensors: the first has shape [max_num_ped,2] and
        contains the sampled coordinates. The second has shape [max_num_ped,
        embedding_size] and contains the output of the linear layer with input
        the sampled coordinates.

    """

    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 - 22
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(std_x)
        std_y = tf.exp(std_y)

        mu = tf.stack([mu_x, mu_y], 1)
        std = tf.stack([std_x, std_y], 1)

        # Sample the coordinates
        dist = tf.distributions.Normal(mu, std)
        coordinates = dist.sample()
        return coordinates, layer_output(coordinates)


def social_sample_position_estimate_stabilized(
    cell_output, coordinates_gt, output_size, layer_output
):
    """Calculate the coordinates in sampling phase. Stabilized version.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Ground truth
        coordinates.
      output_size: int. Dimension of the output size.
      layer_output: tf.layer instance. Layer used for process the new
        coordinates sampled.

    Returns:
      tuple containing two tensors: the first has shape [max_num_ped,2] and
        contains the sampled coordinates. The second has shape [max_num_ped,
        embedding_size] and contains the output of the linear layer with input
        the sampled coordinates.

    """

    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 -> 22
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(tf.abs(std_x))
        std_y = tf.exp(tf.abs(std_y))

        mu = tf.stack([mu_x, mu_y], 1)
        std = tf.stack([std_x, std_y], 1)

        # Sample the coordinates
        dist = tf.distributions.Normal(mu, std)
        coordinates = dist.sample()
        return coordinates, layer_output(coordinates)
