"""Module that defines the classes that provides the input for the classes
defined in the dataset module. Each class load datasets, preprocess them and
create two generators that return sequences or batches of trajectories.

"""
import os
import random
import logging
import numpy as np
import tensorflow as tf


class DataLoader:
    """Data loader class that load the given datasets, preprocess them and create
    two generators that return sequences or batches of trajectories.

    """

    def __init__(
        self,
        data_path,
        datasets,
        delimiter="\t",
        skip=1,
        max_num_ped=100,
        trajectory_size=20,
        batch_size=10,
    ):
        """Constructor of the DataLoader class.

        Args:
          data_path: string. Path to the folder containing the datasets
          datasets: list. List of datasets to use.
          delimiter: string. Delimiter used to separate data inside the
            datasets.
          skip: int or True. If True, the number of frames to skip while making
            the dataset is random. If int, number of frames to skip while making
            the dataset
          max_num_ped: int. Maximum number of pedestrian in a single frame.
          trajectories_size: int. Length of the trajectory (obs_length +
            pred_len).
          batch_size: int. Batch size.

        """
        # Store the list of datasets to load
        self.__datasets = [os.path.join(data_path, dataset) for dataset in datasets]
        logging.debug(
            "Number of dataset will be loaded: {} List of datasets: {}".format(
                len(self.__datasets), self.__datasets
            )
        )

        # Store the batch_size, trajectory_size, the maximum number of
        # pedestrian in a single frame and skip value
        self.batch_size = batch_size
        self.trajectory_size = trajectory_size
        self.max_num_ped = max_num_ped
        self.skip = skip

        if delimiter == "tab":
            delimiter = "\t"
        elif delimiter == "space":
            delimiter = " "

        # Load the datasets and preprocess them
        self.__load_data(delimiter)
        self.__preprocess_data()
        self.__type_and_shape()

    def next_batch(self):
        """Generator method that returns an iterator pointing to the next batch.

        Returns:
            Generator object that has a list of trajectory sequences of size
              batch_size, a list of relative trajectory sequences of size
              batch_size, a list containing the associated grid layer of size
              batch_size, a list with the number of pedestrian in each sequence
              and a list containing the mask for the loss function.

        """
        it = self.next_sequence()
        for batch in range(self.num_batches):
            batch = []
            batch_rel = []
            grid_batch = []
            peds_batch = []
            loss_batch = []

            for size in range(self.batch_size):
                data = next(it)
                batch.append(data[0])
                batch_rel.append(data[1])
                grid_batch.append(data[2])
                peds_batch.append(data[3])
                loss_batch.append(data[4])

            yield batch, batch_rel, grid_batch, peds_batch, loss_batch

    def next_sequence(self):
        """Generator method that returns an iterator pointing to the next sequence.

        Returns:
          Generator object that contains a trajectory sequence, a relative
            trajectory sequence, the associated grid layer, the number of
            pedestrian in the sequence and the mask for the loss function.

        """
        # Iterate through all sequences
        for idx_d, dataset in enumerate(self.__trajectories):
            # Every dataset
            for idx_s, trajectories in enumerate(dataset):
                sequence, grid, loss_mask = self.__get_sequence(trajectories)

                # Create the relative coordinates
                sequence_rel = np.zeros(
                    [self.trajectory_size, self.max_num_ped, 2], float
                )
                sequence_rel[1:] = sequence[1:] - sequence[:-1]
                num_peds = self.__num_peds[idx_d][idx_s]

                yield (sequence, sequence_rel, grid, num_peds, loss_mask)

    def __load_data(self, delimiter):
        """Load the datasets and define the list __frames.

        Load the datasets and define the list __frames wich contains all the
        frames of the datasets. __frames has shape [num_datasets,
        num_frames_dataset, num_peds_frame, 4] where 4 is frameID, pedID,
        x and y.

        Args:
          delimiter: string. Delimiter used to separate data inside the
            datasets.

        """
        # List that contains all the frames of the datasets. Each dataset is a
        # list of frames of shape (num_peds, (frameID, pedID, x and y))
        self.__frames = []

        for dataset_path in self.__datasets:
            # Load the dataset. Each line is formed by frameID, pedID, x, y
            dataset = np.loadtxt(dataset_path, delimiter=delimiter)
            # Get the frames in dataset
            num_frames = np.unique(dataset[:, 0])
            # Initialize the array of frames for the current dataset
            frames_dataset = []

            # For each frame add to frames_dataset the pedestrian that appears
            # in the current frame
            for frame in num_frames:
                # Get the pedestrians
                frame = dataset[dataset[:, 0] == frame, :]
                frames_dataset.append(frame)

            self.__frames.append(frames_dataset)

    def __preprocess_data(self):
        """Preprocess the datasets and define the number of sequences and batches.

        The method iterates on __frames saving on the list __trajectories only
        the trajectories with length trajectory_size.

        """
        # Keep only the trajectories trajectory_size long
        self.__trajectories = []
        self.__num_peds = []
        self.num_sequences = 0

        for dataset in self.__frames:
            # Initialize the array of trajectories for the current dataset.
            trajectories = []
            num_peds = []
            frame_size = len(dataset)
            i = 0

            # Each trajectory contains only frames of a dataset
            while i + self.trajectory_size < frame_size:
                sequence = dataset[i : i + self.trajectory_size]
                # Get the pedestrians in the first frame
                peds = np.unique(sequence[0][:, 1])
                # Check if the trajectory of pedestrian is long enough.
                sequence = np.concatenate(sequence, axis=0)
                traj_frame = []
                for ped in peds:
                    # Get the frames where ped appear
                    frames = sequence[sequence[:, 1] == ped]
                    # Check the trajectory is long enough
                    if frames.shape[0] == self.trajectory_size:
                        traj_frame.append(frames)
                # If no trajectory is long enough traj_frame is empty. Otherwise
                if traj_frame:
                    trajectories_frame, peds_frame = self.__create_sequence(
                        traj_frame, sequence
                    )
                    trajectories.append(trajectories_frame)
                    num_peds.append(peds_frame)
                    self.num_sequences += 1
                # If skip is True, update the index with a random value
                if self.skip is True:
                    i += random.randint(0, self.trajectory_size)
                else:
                    i += self.skip

            self.__trajectories.append(trajectories)
            self.__num_peds.append(num_peds)

        # num_batches counts only full batches. It discards the remaining
        # sequences
        self.num_batches = int(self.num_sequences / self.batch_size)
        logging.info("There are {} sequences in loader".format(self.num_sequences))
        logging.info("There are {} batches in loader".format(self.num_batches))

    def __get_sequence(self, trajectories):
        """Returns a tuple containing a trajectory sequence and the grid mask.

        Args:
          trajectories: list of numpy array. Each array is a trajectory.

        Returns:
          tuple containing a numpy array with shape [trajectory_size,
            max_num_ped, 2] that contains the trajectories, a numpy array with
            shape [trajectory_size, max_num_ped, max_num_ped] that is the grid
            layer and a numpy array with shape [trajectory_size, max_num_ped]
            that is the mask for the loss function.

        """
        num_peds_sequence = len(trajectories)
        sequence = np.zeros((self.max_num_ped, self.trajectory_size, 2))
        grid = np.zeros((self.max_num_ped, self.trajectory_size), dtype=bool)

        sequence[:num_peds_sequence] = trajectories[:, :, [2, 3]]

        # Create the grid layer. Set to True only the pedestrians that are in
        # the sequence. A pedestrian is in the sequence if its frameID is not 0
        grid[:num_peds_sequence] = trajectories[:, :, 0]
        # Create the mask for the loss function
        loss_mask = grid
        # Create a grid for all the pedestrians
        grid = np.tile(grid, (self.max_num_ped, 1, 1))
        # Grid layer ignores the pedestrian itself
        for ped in range(num_peds_sequence):
            grid[ped, ped] = False

        # Change shape of the arrays. From [max_num_ped, trajectory_size] to
        # [trajectory_size, max_num_ped]
        sequence_moved = np.moveaxis(sequence, 1, 0)
        grid_moved = np.moveaxis(grid, 2, 0)
        loss_moved = np.moveaxis(loss_mask, 1, 0)

        return sequence_moved, grid_moved, loss_moved

    def __create_sequence(self, trajectories_full, sequence):
        """Create an array with the trajectories contained in a dataset slice.

        Args:
          trajectories_full: list that contains the trajectories long
            trajectory_size of the dataset slice.
          sequence: list that contains the remaining trajectories of the dataset
            slice.

        Returns:
          tuple containing the ndarray with the trajectories of the dataset
            slice and the number of pedestrians thate are trajectory_size long
            in the dataset slice. In the first positions of the ndarray there
            are the trajectories long enough. The shape of the ndarray is
            [peds_sequence, trajectory_size, 4].

        """
        trajectories_full = np.array(trajectories_full)
        peds_sequence = np.unique(sequence[:, 1])
        peds_trajectories = np.unique(trajectories_full[:, :, 1])
        frames_id = np.unique(sequence[:, 0])
        # Create the array that will contain the trajectories
        trajectories = np.zeros((len(peds_sequence), self.trajectory_size, 4))

        # Copy trajectories_full in the first len(peds_trajectories) rows
        trajectories[: len(peds_trajectories)] = trajectories_full
        # Remove the peds that are in peds_trajectories
        peds_sequence = np.delete(
            peds_sequence, np.searchsorted(peds_sequence, peds_trajectories)
        )
        # Create a lookup table with the frames id and their position in the
        # sequence
        lookup_frames = {}
        for i, frame in enumerate(frames_id):
            lookup_frames[frame] = i

        # Add the remaining peds
        for i, ped in enumerate(peds_sequence, len(peds_trajectories)):
            # Get the indexes where the pedsID is equal to ped
            positions = np.where(sequence[:, 1] == ped)[0]
            # Use the lookup table to find out where the pedestrian trajectory
            # begins and end in the sequence
            start = lookup_frames[sequence[positions][0, 0]]
            end = lookup_frames[sequence[positions][-1, 0]] + 1
            # Copy the pedestrian trajectory inside the sequence
            trajectories[i, start:end] = sequence[positions]

        return trajectories, len(peds_trajectories)

    def __type_and_shape(self):
        """Define the type and the shape of the arrays that tensorflow will use"""
        self.output_types = (tf.float32, tf.float32, tf.bool, tf.int32, tf.int32)
        self.shape = (
            tf.TensorShape([self.trajectory_size, self.max_num_ped, 2]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, 2]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, self.max_num_ped]),
            tf.TensorShape([]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped]),
        )
