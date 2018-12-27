"""Module that defines the SocialLSTM model."""
import tensorflow as tf
import trajectory_decoders


class SocialModel:
    """SocialModel defines the model described in the Social LSTM paper."""

    def __init__(
        self,
        dataset,
        helper,
        position_estimate,
        loss_function,
        hparams,
        pooling_module=None,
    ):
        """Constructor of the SocialModel class.

        Args:
          dataset: A TrajectoriesDataset instance.
          helper: A coordinates helper function.
          position_estimate: A position_estimate function.
          loss_function: a loss funtcion.
          hparams: An HParams instance.
          pooling_module: A pooling module function.

        """
        # Create the tensor for input_data of shape
        # [max_num_ped, trajectory_size, 2]
        self.input_data = dataset.tensors[0]
        # Create the tensor for the ped mask of shape [max_num_ped, max_num_ped,
        # trajectory_size]
        self.peds_mask = dataset.tensors[1]
        # Create the tensor for num_peds_frame
        self.num_peds_frame = dataset.tensors[2]
        # Create the tensor for all pedestrians of shape
        # [max_num_ped, trajectory_size,2]
        self.all_peds = dataset.tensors[3]
        # Create the for the ped mask of shape
        # [max_num_ped, max_num_ped, trajectory_size]
        self.all_peds_mask = dataset.tensors[4]

        # Store the parameters
        # In training phase the list contains the values to minimize. In
        # sampling phase it has the coordinates predicted
        self.new_coordinates = []
        # The predicted coordinates processed by the linear layer in sampling
        # phase
        new_coordinates_processed = None
        # Output (or hidden states) of the LSTMs
        cell_output = tf.zeros([hparams.max_num_ped, hparams.lstm_size], tf.float32)

        # Output size
        output_size = 5
        # Trajectory size
        trajectory_size = hparams.obsLen + hparams.predLen

        # Counter for the adaptive learning rate. Counts the number of batch
        # processed.
        global_step = tf.Variable(0, trainable=False)

        # Define the LSTM with dimension lstm_size
        with tf.variable_scope("LSTM"):
            self.cell = tf.nn.rnn_cell.LSTMCell(hparams.lstm_size, name="Cell")

            # Define the states of the LSTMs. zero_state returns a tensor of
            # shape [max_num_ped, state_size]
            with tf.name_scope("States"):
                self.cell_states = self.cell.zero_state(hparams.max_num_ped, tf.float32)

        # Define the layer with ReLu used for processing the coordinates
        self.coordinates_layer = tf.layers.Dense(
            hparams.embedding_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="Coordinates/Layer",
        )

        # Define the layer with ReLu used as output_layer for the decoder
        self.output_layer = tf.layers.Dense(
            output_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="Position_estimation/Layer",
        )

        # Define the SocialTrajectoryDecoder.
        decoder = trajectory_decoders.SocialDecoder(
            self.cell,
            hparams.max_num_ped,
            helper,
            pooling_module=pooling_module,
            output_layer=self.output_layer,
        )

        # Decode the coordinates
        for frame in range(trajectory_size - 1):
            # Processing the coordinates
            self.coordinates_preprocessed = self.coordinates_layer(
                self.input_data[:, frame]
            )
            all_peds_preprocessed = self.coordinates_layer(self.all_peds[:, frame])

            # Initialize the decoder passing the real coordinates, the
            # coordinates that the model has predicted and the states of the
            # LSTMs. Which coordinates the model will use will be decided by the
            # helper function
            decoder.initialize(
                frame,
                self.coordinates_preprocessed,
                new_coordinates_processed,
                self.cell_states,
                all_peds_preprocessed,
                hidden_states=cell_output,
                peds_mask=self.peds_mask[:, :, frame],
                all_peds_mask=self.all_peds_mask[:, :, frame],
            )
            # compute_pass returns a tuple of two tensors. cell_output are the
            # output of the self.cell with shape [max_num_ped , output_size] and
            # cell_states are the states with shape
            # [max_num_ped, LSTMStateTuple()]
            cell_output, self.cell_states, layered_output = decoder.step()

            # Compute the new coordinates
            new_coordinates, new_coordinates_processed = position_estimate(
                layered_output,
                self.input_data[:, frame + 1],
                output_size,
                self.coordinates_layer,
            )

            # Append new_coordinates
            self.new_coordinates.append(new_coordinates)

        # self.new_coordinates has shape [trajectory_size - 1, max_num_ped]
        self.new_coordinates = tf.stack(self.new_coordinates)

        with tf.variable_scope("Calculate_loss"):
            self.loss = loss_function(
                self.new_coordinates[-hparams.predLen :, : self.num_peds_frame]
            )
            self.loss = tf.div(self.loss, tf.cast(self.num_peds_frame, tf.float32))

        # Add weights regularization
        tvars = tf.trainable_variables()
        l2_loss = (
            tf.add_n([tf.nn.l2_loss(v) for v in tvars if "bias" not in v.name])
            * hparams.l2_norm
        )
        self.loss = self.loss + l2_loss

        # Step epoch learning rate decay
        learning_rate = tf.train.exponential_decay(
            hparams.learning_rate, global_step, hparams.lr_steps, hparams.lr_decay
        )

        # Define the RMSProp optimizer
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=hparams.opt_decay,
            momentum=hparams.opt_momentum,
            centered=hparams.centered,
        )
        # Global norm clipping
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        clipped, _ = tf.clip_by_global_norm(gradients, hparams.clip_norm)
        self.trainOp = optimizer.apply_gradients(
            zip(clipped, variables), global_step=global_step
        )
