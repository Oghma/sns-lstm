import tensorflow as tf


def social_loss_function(coordinates):
    """Calculate the negative log-Likelihood loss.

    Args:
      coordinates: tensor of shape [trajectory_size]. The values calculated from
        the train_position_estimate function.

    Returns:
      The negative log-Likelihood loss.

    """
    # For numerical stability
    epsilon = 1e-20

    return -tf.reduce_sum(tf.log(tf.maximum(coordinates, epsilon)))
