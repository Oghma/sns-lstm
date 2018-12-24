"""Module that defines the decoders for the trajectories. Each decoder has an
initialize method that store the variables for the step method and a step method
that compute a step of decoding the coordinates

"""
import tensorflow as tf


class SocialDecoder:
    """SocialDecoder defines the decoder used in the SocialLSTM paper.

    The class has two methods: initialize and step. Initialize stores the
    variables needed for computing a step of decoding. Step computes a pass of
    decoding.

    """

    def __init__(
        self,
        cell,
        max_num_ped,
        coordinates_helper,
        pooling_module=None,
        output_layer=None,
    ):
        """Constructor of the SocialDecoder class.

        Args:
          cell: An RNNCell instance.
          max_num_ped: int. The maximum number of pedestrian in each frame of
            the sequence.
          coordinates_helper: helper function. Function used to decide which
            coordinates to use: ground truth or predicted.
          pooling_module: A pooling module instance.
          output_layer: tf.layer instance. Layer used for process the output of
          the cell.

        """
        self.__cell = cell
        self.max_num_ped = max_num_ped
        self.__pooling_module = pooling_module
        self.__output_layer = output_layer
        self.__coordinates_helper = coordinates_helper

    def step(self, states=None):
        """Compute a pass of decoding.

        Args:
          states: tensor of shape [max_num_ped, embedding_size]. Cell states
          used for the social_module

        Returns:
          tuple containing the outuput and the cell states of the cell instance.

        """
        # If pooling_module is not None, apply the pooling_module
        if self.__pooling_module is not None:
            cell_input = tf.concat([self.__input_pass, self.__pooling], 1)
        else:
            cell_input = self.__input_pass

        # Compute a pass
        cell_output, new_state = self.__cell(cell_input, self.__cell_states)

        # If output_layer is not None, apply the output layer
        if self.__output_layer is not None:
            layered_output = self.__output_layer(cell_output)
        else:
            layered_output = cell_output

        return (cell_output, new_state, layered_output)

    def initialize(
        self,
        step,
        input_pass_gt,
        input_pass,
        cell_states,
        all_peds,
        hidden_states=None,
        peds_mask=None,
        all_peds_mask=None,
    ):
        """Store the variables needed for computing one pass of decoding.

        Args:
          step: int. The step to be calculated.
          input_pass_gt: tensor of shape [max_num_ped, 2]. Ground truth
            coordinates.
          input_pass: tensor of shape [max_num_ped, 2]. Predicted coordinates.
          cell_states: LSTMStateTuple instance. The cell states of the rnn
            network.

        """
        self.__cell_states = cell_states
        self.__hidden_states = hidden_states
        self.__peds_mask = peds_mask
        self.__input_pass = self.__coordinates_helper(step, input_pass_gt, input_pass)

        if self.__pooling_module is not None:
            self.__pooling = self.__pooling_module(
                self.__input_pass,
                self.__input_pass,
                self.__hidden_states,
                self.__peds_mask,
                all_peds,
                all_peds_mask,
            )
