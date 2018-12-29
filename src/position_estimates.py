"""Module that defines the functions for the position estimation"""
import math
import tensorflow as tf


def social_train_position_estimate(cell_output, output_size, coordinates_gt):
    """Calculate the probability density function in training phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      output_size: int. Dimension of the output size.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Ground truth
        coordinates.

    Returns:
      tensor of shape [max_num_ped, 2] that contains the pdf.

    """
    # Calculate the probability density function on Graves (2013) equations.
    # Assume a bivariate Gaussian distribution.
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
        return tf.div(n_num, n_den)


def social_sample_position_estimate(cell_output, output_size):
    """Calculate the coordinates in sampling phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      output_size: int. Dimension of the output size.

    Returns:
      tensor of shape [max_num_ped, 2] that contains the sampled coordinates.

    """

    # Calculate the new coordinates based on Graves (2013) equations. Assume a
    # bivariate Gaussian distribution.
    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 - 22 from Graves
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(std_x)
        std_y = tf.exp(std_y)
        rho = tf.tanh(rho)

        # Kaiser-Dickman algorithm (Kaiser & Dickman, 1962)
        # Generate two sample X1, X2 from the standard normal distribution (mu =
        # 0, sigma = 1)
        normal_coords = tf.random.normal(tf.TensorShape([mu_x.shape[0], 2]))
        # Generate the correlation.
        # correlation = rho * X1 + sqrt(1 - pow(rho)) * X2
        correlation = (
            rho * normal_coords[:, 0]
            + tf.sqrt(1 - tf.square(rho)) * normal_coords[:, 1]
        )

        # Define the two coordinates correlated
        # Y1 = mu_x + sigma_x * X1
        # Y2 = mu_y + sigma_y * correlation
        coords_x = mu_x + std_x * normal_coords[:, 0]
        coords_y = mu_y + std_y * correlation

        coordinates = tf.stack([coords_x, coords_y], 1)
        return coordinates
