"""Module that defines the SocialLSTM model."""
import logging
import tensorflow as tf

import losses
import pooling_layers
import position_estimates
import coordinates_helpers

TRAIN = "TRAIN"
SAMPLE = "SAMPLE"


class SocialModel:
    """SocialModel defines the model described in the Social LSTM paper."""

    def __init__(self, dataset, hparams, phase=TRAIN):
        """Constructor of the SocialModel class.

        Args:
          dataset: A TrajectoriesDataset instance.
          hparams: An HParams instance.
          phase: string. Train or sample phase

        """
        # Create the tensor for the pedestrians of shape
        # [trajectory_size, max_num_ped, 2]
        self.pedestrians_coordinates = dataset.tensors[0]
        # Create the tensor for the pedestrian relative coordinates of shape
        # [trajectory_size, max_num_ped, 2]
        pedestrians_coordinates_rel = dataset.tensors[1]
        # Create the tensor for the ped mask of shape
        # [trajectory_size, max_num_ped, max_num_ped]
        pedestrians_mask = dataset.tensors[2]
        # Create the tensor for num_peds_frame
        self.num_peds_frame = dataset.tensors[3]
        # Create the tensor for the loss mask of shape
        # [trajectory_size, max_num_ped]
        loss_mask = dataset.tensors[4]
        # Create the tensor for the navigation map of shape
        # [navigation_width, navigation_height]
        navigation_map = dataset.tensors[5]
        # Create the tensor for the upper left-most point in the dataset
        top_left_dataset = dataset.tensors[6]
        # Create the tensor for the semantic map of shape
        # [num_points, 2 + num_labels]
        semantic_map = dataset.tensors[7]
        # Create the tensor for the homography matrix of shape [3,3]
        homography = dataset.tensors[8]
        # Create the tensor for the pedestrian relative coordinates of shape
        # [max_num_ped, 2]
        new_pedestrians_coordinates_rel = tf.zeros([hparams.maxNumPed, 2])

        # Store the parameters
        # Output size of the linear layer
        output_size = 5
        trajectory_size = hparams.obsLen + hparams.predLen
        # Contain the predicted coordinates or the pdf of the last frame computed
        new_pedestrians_coordinates = tf.TensorArray(
            dtype=tf.float32, size=trajectory_size, clear_after_read=False
        )

        # Counter for the adaptive learning rate. Counts the number of batch
        # processed.
        global_step = tf.Variable(0, trainable=False, name="Global_step")

        # ============================ BUILD MODEL ============================

        # Create the helper class
        logging.info("Creating the social helper")
        if phase == TRAIN:
            helper = coordinates_helpers.train_helper
        elif phase == SAMPLE:
            helper = coordinates_helpers.sample_helper(hparams.obsLen)

        # Create the pooling layer
        pooling_module = None
        if isinstance(hparams.poolingModule, list):
            logging.info(
                "Creating the combined pooling: {}".format(hparams.poolingModule)
            )
            pooling_module = pooling_layers.CombinedPooling(hparams).pooling

        elif hparams.poolingModule == "social":
            logging.info("Creating the {} pooling".format(hparams.poolingModule))
            pooling_module = pooling_layers.SocialPooling(hparams).pooling

        elif hparams.poolingModule == "occupancy":
            logging.info("Creating the {} pooling".format(hparams.poolingModule))
            pooling_module = pooling_layers.OccupancyPooling(hparams).pooling

        elif hparams.poolingModule == "navigation":
            logging.info("Creating the {} pooling".format(hparams.poolingModule))
            pooling_module = pooling_layers.NavigationPooling(hparams).pooling
        elif hparams.poolingModule == "semantic":
            logging.info("Creating the {} pooling".format(hparams.poolingModule))
            pooling_module = pooling_layers.SemanticPooling(hparams).pooling

        # Create the position estimates functions
        logging.info("Creating the social position estimate function")
        if phase == TRAIN:
            position_estimate = position_estimates.social_train_position_estimate
        elif phase == SAMPLE:
            position_estimate = position_estimates.social_sample_position_estimate

        # Create the loss function
        logging.info("Creating the social loss function")
        loss_function = losses.social_loss_function

        # Dropout rate
        if phase == SAMPLE:
            hparams.set_hparam("keepProb", 1.0)

        # ============================ MODEL LAYERS ============================

        # Define the LSTM with dimension rnn_size
        with tf.variable_scope("LSTM"):
            cell = tf.nn.rnn_cell.LSTMCell(hparams.rnnSize, name="Cell")
            # Apply dropout
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=hparams.keepProb
            )
            # Output (or hidden states) of the LSTMs
            cell_output = tf.zeros(
                [hparams.maxNumPed, hparams.rnnSize], tf.float32, name="Output"
            )
            # Define the states of the LSTMs. zero_state returns a named tuple
            # with two tensors of shape [max_num_ped, state_size]
            with tf.name_scope("States"):
                cell_states = cell.zero_state(hparams.maxNumPed, tf.float32)

        # Define the layer with ReLu used for processing the coordinates
        coordinates_layer = tf.layers.Dense(
            hparams.embeddingSize,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="Coordinates/Layer",
        )

        # Define the layer with ReLu used as output_layer for the decoder
        output_layer = tf.layers.Dense(
            output_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="Position_estimation/Layer",
        )

        # ============================ LOOP FUNCTIONS ===========================

        frame = tf.constant(0)

        # If phase is TRAIN, new_pedestrians_coordinates contains the pdf and it
        # has shape [trajectory_size, max_num_ped]. If phase is SAMPLE
        # new_pedestrians_coordinates contains the coordinates predicted and it
        # has shape [trajectory_size, max_num_ped, 2]
        if phase == TRAIN:
            new_pedestrians_coordinates = new_pedestrians_coordinates.write(
                0, tf.zeros(hparams.maxNumPed)
            )
        elif phase == SAMPLE:
            new_pedestrians_coordinates = new_pedestrians_coordinates.write(
                0, self.pedestrians_coordinates[0]
            )

        def cond(frame, *args):
            return frame < (trajectory_size - 1)

        def body(
            frame,
            new_pedestrians_coordinates,
            new_pedestrians_coordinates_rel,
            cell_output,
            cell_states,
        ):
            # Processing the coordinates. Apply the liner layer with relu
            current_coordinates, current_coordinates_rel = helper(
                frame,
                self.pedestrians_coordinates[frame],
                pedestrians_coordinates_rel[frame],
                new_pedestrians_coordinates.read(frame),
                new_pedestrians_coordinates_rel,
            )
            pedestrians_coordinates_preprocessed = coordinates_layer(
                current_coordinates_rel
            )
            # Apply dropout
            pedestrians_coordinates_preprocessed = tf.nn.dropout(
                pedestrians_coordinates_preprocessed, hparams.keepProb
            )

            # If pooling_module is not None, add the pooling layer
            if pooling_module is not None:
                pooling_output = pooling_module(
                    current_coordinates,
                    states=cell_output,
                    peds_mask=pedestrians_mask[frame],
                    navigation_map=navigation_map,
                    top_left_dataset=top_left_dataset,
                    semantic_map=semantic_map,
                    H=homography,
                )
                cell_input = tf.concat(
                    [pedestrians_coordinates_preprocessed, pooling_output],
                    1,
                    name="Rnn_input",
                )
            else:
                cell_input = pedestrians_coordinates_preprocessed

            # Compute a pass
            cell_output, cell_states = cell(cell_input, cell_states)

            # Apply the linear layer to the cell output
            layered_output = output_layer(cell_output)
            # Apply dropout
            layered_output = tf.nn.dropout(layered_output, hparams.keepProb)

            # Compute the new coordinates or the pdf
            if phase == TRAIN:
                coordinates_predicted = position_estimate(
                    layered_output, output_size, pedestrians_coordinates_rel[frame + 1]
                )
            elif phase == SAMPLE:
                new_pedestrians_coordinates_rel = position_estimate(
                    layered_output, output_size
                )
                coordinates_predicted = (
                    new_pedestrians_coordinates.read(frame)
                    + new_pedestrians_coordinates_rel
                )

            # Append new_coordinates
            new_pedestrians_coordinates = new_pedestrians_coordinates.write(
                frame + 1, coordinates_predicted
            )
            return (
                frame + 1,
                new_pedestrians_coordinates,
                new_pedestrians_coordinates_rel,
                cell_output,
                cell_states,
            )

        # Decode the coordinates
        _, new_pedestrians_coordinates, _, _, _ = tf.while_loop(
            cond,
            body,
            loop_vars=[
                frame,
                new_pedestrians_coordinates,
                new_pedestrians_coordinates_rel,
                cell_output,
                cell_states,
            ],
        )

        # In training phase the list contains the values to minimize. In
        # sampling phase it has the coordinates predicted. Tensor has shape
        # [trajectory_size, max_num_ped, 2]
        self.new_pedestrians_coordinates = new_pedestrians_coordinates.stack(
            "new_coordinates"
        )

        if phase == TRAIN:
            with tf.variable_scope("Calculate_loss"):
                # boolean_mask produces a warning because a sparse IndexedSlices
                # is implicitly converted to a dense tensor. This typically
                # happens when one operation (usually tf.gather())
                # backpropagates a sparse gradient, but the op that receives it
                # does not have a specialized gradient function that can handle
                # sparse gradients. As a result, TensorFlow automatically
                # densifies the tf.IndexedSlices, which can have a devastating
                # effect on performance if the tensor is large. To fix the
                # problem we use dynamic_partition. The values for the loss
                # function are in the second list
                loss_values = tf.dynamic_partition(
                    self.new_pedestrians_coordinates[-hparams.predLen :],
                    loss_mask[-hparams.predLen :],
                    2,
                )
                self.loss = loss_function(loss_values[1])
                self.loss = tf.div(self.loss, tf.cast(self.num_peds_frame, tf.float32))

            # Add weights regularization
            tvars = tf.trainable_variables()
            l2_loss = (
                tf.add_n([tf.nn.l2_loss(v) for v in tvars if "bias" not in v.name])
                * hparams.l2Rate
            )
            self.loss = self.loss + l2_loss

            # Step epoch learning rate decay
            learning_rate = tf.train.exponential_decay(
                hparams.learningRate,
                global_step,
                hparams.learningRateSteps,
                hparams.learningRateDecay,
            )

            # Define the RMSProp optimizer
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=hparams.optimizerDecay,
                momentum=hparams.optimizerMomentum,
                centered=hparams.centered,
            )
            # Global norm clipping
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            clipped, _ = tf.clip_by_global_norm(gradients, hparams.clippingRatio)
            self.train_optimizer = optimizer.apply_gradients(
                zip(clipped, variables), global_step=global_step
            )
