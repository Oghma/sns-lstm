"""Module that defines the coordinates helper. An helper provides the correct
coordinates according to the traininig or sampling phase and the time-step."""
import tensorflow as tf


def train_helper(step, coordinates_gt, coordinates_gt_rel, *args):
    """Helper used in training phase. Returns the ground truth coordinates.

    Args:
      step: int. The current time-step.
      coordinates_gt: tensor of shape [max_num_ped, 2]. The ground truth
        coordinates.
      coordinates_gt_rel: tensor of shape [max_num_ped, 2]. The ground truth
        relative coordinates.

    Returns:
      In training phase always returns the ground truth coordinates.

    """
    return coordinates_gt, coordinates_gt_rel


def sample_helper(obs_len):
    """Helper used in sampling phase. Returns a function that returns the ground
    truth coordinates or the predicted coordinates based on the value of step.

    In sampling phase, if the step variable is less of the obs_len, the helper
    returns the ground truth coordinates. Otherwise returns the predicted
    coordinates.

    Args:
      obs_len: int. The number of time-step to be observed.

    Returns:
      A function that receives in input the time-step, the ground truth
        coordinates and the predicted coordinates and returns the right
        coordinates according to the time-step.

    """

    def helper(
        step,
        coordinates_gt,
        coordinates_gt_rel,
        coordinates_predicted,
        coordinates_predicted_rel,
    ):
        return tf.cond(
            step >= obs_len,
            lambda: (coordinates_predicted, coordinates_predicted_rel),
            lambda: (coordinates_gt, coordinates_gt_rel),
        )

    return helper
