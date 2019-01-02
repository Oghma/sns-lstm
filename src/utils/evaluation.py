"""Module that defines the metrics used for evaluation."""

import tensorflow as tf


def average_displacement_error(coordinates_predicted, coordinates_gt, num_peds):
    """Function that calculates the average displacement error.

    The formula is
    sqrt(sum((coordinates_gt - coordinates_predicted)^2)) / (num_peds * pred_length)

    Args:
      coordinates_predicted: tensor of shape [pred_length, max_num_ped, 2].
        Tensor that contains the coordinates predicted from the model.
      coordinates_gt: tensor of shape [pred_length, max_num_ped, 2]. Tensor that
        contains the ground truth coordinates.
      num_peds: tensor with type tf.Int32 . Number of pedestrians that are in
        the sequence.

    Returns:
      tensor with type tf.float32 containing the average displacement error of
        the pedestrians that are in the sequence.

    """
    i = tf.constant(0)
    ade = tf.constant(0, tf.float32)

    cond = lambda i, ade: tf.less(i, num_peds)

    def body(i, ade):
        ade_ped = coordinates_gt[:, i] - coordinates_predicted[:, i]
        ade_ped = tf.norm(ade_ped)
        return tf.add(i, 1), tf.add(ade, ade_ped)

    _, ade = tf.while_loop(cond, body, [i, ade])
    return ade / tf.cast(num_peds * coordinates_gt.shape[0], tf.float32)


def final_displacement_error(coordinates_predicted, coordinates_gt, num_peds):
    """Function that calculates the final displacement error.

    The formula is
    sqrt(sum((coordinates_gt - coordinates_predicted)^2)) / num_peds

    Args:
      coordinates_predicted: tensor of shape [max_num_ped, 2]. Tensor that
        contains the coordinates predicted from the model.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Tensor that contains the
        ground truth coordinates.
      num_peds: tensor with type tf.Int32 . Number of pedestrians that are in
        the sequence.

    Returns:
      tensor with type tf.float32 containing the final displacement error of the
        pedestrians that are in the sequence.

    """
    i = tf.constant(0)
    fde = tf.constant(0, tf.float32)

    cond = lambda i, fde: tf.less(i, num_peds)

    def body(i, fde):
        fde_ped = coordinates_gt[i] - coordinates_predicted[i]
        fde_ped = tf.norm(fde_ped)
        return tf.add(i, 1), tf.add(fde, fde_ped)

    _, fde = tf.while_loop(cond, body, [i, fde])
    return fde / tf.cast(num_peds, tf.float32)
