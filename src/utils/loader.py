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
            Generator object that contains a list containing sequences of
              trajectories of size batch_size, a list containing sequences of
              relative trajectories of size batch_size, a list containing the
              associated grid layer of size batch_size a list containing the
              number of pedestrian in the sequences, a list containing all
              pedestrians in the sequence and a list containing the grid layer
              with all pedestrians.

        """
        it = self.next_sequence()
        for batch in range(self.num_batches):
            batch = []
            batch_rel = []
            grid_batch = []
            peds_in_batch = []
            all_peds = []
            all_peds_mask = []

            for size in range(self.batch_size):
                data = next(it)
                batch.append(data[0])
                batch_rel.append(data[1])
                grid_batch.append(data[2])
                peds_in_batch.append(data[3])
                all_peds.append(data[4])
                all_peds_mask.append(data[5])
            yield batch, batch_rel, grid_batch, peds_in_batch, all_peds, all_peds_mask

    def next_sequence(self):
        """Generator method that returns an iterator pointing to the next sequence.

        Returns:
          Generator object that contains a sequence of trajectories, a sequence
            of relative trajectories, the associated grid layer the number of
            pedestrian in the sequence, a list containing all pedestrians in the
            sequence and a list containing the grid layer with all pedestrians.

        """
        # Iterate through all sequences
        for idx_d, dataset in enumerate(self.__trajectories):
            # Every dataset
            for idx_s, trajectories in enumerate(dataset):
                all_peds = self.__trajectories_all_peds[idx_d][idx_s]
                all_peds_moved = np.moveaxis(all_peds[:, :, [2, 3]], 1, 0)
                sequence_rel = np.zeros(
                    [self.trajectory_size, self.max_num_ped, 2], float
                )
                sequence, grid, num_peds, all_peds_mask = self.__get_sequence(
                    trajectories, all_peds[:, :, 1]
                )
                sequence_rel[1:] = sequence[1:] - sequence[:-1]
                yield (
                    sequence,
                    sequence_rel,
                    grid,
                    num_peds,
                    all_peds_moved,
                    all_peds_mask,
                )

    def __load_data(self, delimiter):
        """Load the datasets and define the list __frames.

        Load the datasets and define the list __frames wich contains all the
        frames of the datasets. __frames has shape [num_datasets,
        num_frames_in_dataset, num_peds_in_frame, 4] where 4 is frameID, pedID,
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
        the trajectories with length trajectory_size. The list
        __trajectories_all_peds preserves all the trajectories inside the
        sequence.

        """
        # Keep only the trajectories trajectory_size long
        self.__trajectories = []
        self.__trajectories_all_peds = []
        self.num_sequences = 0

        for dataset in self.__frames:
            # Initialize the array of trajectories for the current dataset.
            trajectories = []
            trajectories_all_peds = []
            frame_size = len(dataset)
            i = 0

            # Each trajectory contains only frames of a dataset
            while i + self.trajectory_size < frame_size:
                sequences = dataset[i : i + self.trajectory_size]
                # Get the pedestrians in the first frame
                peds = np.unique(sequences[0][:, 1])
                # Check if the trajectory of pedestrian is long enough.
                sequence = np.concatenate(sequences, axis=0)
                traj_frame = []
                for ped in peds:
                    # Get the frames where ped appear
                    frames = sequence[sequence[:, 1] == ped, :]
                    # Check the trajectory is long enough
                    if frames.shape[0] == self.trajectory_size:
                        traj_frame.append(frames)
                # If no trajectory is long enough traj_frame is empty. Otherwise
                if traj_frame:
                    trajectories.append(traj_frame)
                    trajectories_all_peds.append(self.__all_peds(sequences))
                    self.num_sequences += 1
                # If skip is True, update the index with a random value
                if self.skip is True:
                    i += random.randint(0, self.trajectory_size)
                else:
                    i += self.skip
            self.__trajectories.append(trajectories)
            self.__trajectories_all_peds.append(trajectories_all_peds)

        # num_batches counts only full batches. It discards the remaining
        # sequences
        self.num_batches = int(self.num_sequences / self.batch_size)
        logging.info("There are {} sequences in loader".format(self.num_sequences))
        logging.info("There are {} batches in loader".format(self.num_batches))

    def __get_sequence(self, trajectories, all_peds):
        """Returns a sequence of trajectories of shape [trajectory_size, maxNumPed, 2]

        Args:
          trajectories: list of numpy array. Each array is a trajectory.
          all_peds: numpy array of shape [max_num_ped, trajectory_size]. Array
            that contains the ID of all pedestrians.

        Returns:
          tuple containing a numpy array with shape [trajectory_size,
            max_num_ped, 2] that contains all the trajectories, a numpy array
            with shape [trajectory_size, max_num_ped, max_num_ped] that is the
            grid layer, the number of pedestrian in the sequence and a numpy
            array with shape [trajectory_size, max_num_ped, max_num_ped] that is
            the grid layer of all pedestrians.

        """
        sequence = np.zeros((self.max_num_ped, self.trajectory_size, 2))
        grid = np.zeros(
            (self.max_num_ped, self.max_num_ped, self.trajectory_size), dtype=bool
        )

        num_peds_in_sequence = len(trajectories)
        peds_in_sequence = map(lambda x: int(x[0, 1]), trajectories)
        all_peds_mask = np.repeat(
            np.expand_dims(all_peds, axis=0), self.max_num_ped, axis=0
        )

        for index, trajectory in enumerate(trajectories):
            sequence[index] = trajectory[:, [2, 3]]

        # Create the grid layer. Set to True only the pedestrians that are in
        # the sequence
        grid[:num_peds_in_sequence, :num_peds_in_sequence] = True
        # Grid layer ignores the pedestrian itself
        for ped in range(num_peds_in_sequence):
            grid[ped, ped] = False

        # Create the all_peds_mask
        for idx, ped in enumerate(peds_in_sequence):
            all_peds_mask[idx, all_peds_mask[idx] == ped + 1] = 0
        all_peds_mask = all_peds_mask.astype(bool, copy=False)

        # Chane shape of the arrays. From [max_num_ped, trajectory_size] to
        # [trajectory_size, max_num_ped]
        sequence_moved = np.moveaxis(sequence, 1, 0)
        grid_moved = np.moveaxis(grid, 2, 0)
        all_peds_mask_moved = np.moveaxis(all_peds_mask, 2, 0)

        return sequence_moved, grid_moved, num_peds_in_sequence, all_peds_mask_moved

    def __all_peds(self, sequences):
        """Create a ndarray containing the coordinates of all pedestrians in the
        sequence.

        Args:
          sequences: list of ndarray containing the pedestrian in a single frame.

        Returns:
          ndarray of shape [max_num_ped, trajectory_size, 2] containing the
            coordinates of all pedestrians in the sequence.

        """
        sequence = np.zeros((self.max_num_ped, self.trajectory_size, 4))
        for frame in range(self.trajectory_size):
            peds = len(sequences[frame])
            sequence[:peds, frame] = sequences[frame]
            # Some pedestrains have ID 0. Add 1 to start with ID 1
            sequence[:peds, frame, 1] += 1
        return sequence

    def __type_and_shape(self):
        """Define the type and the shape of the arrays that tensorflow will use"""
        self.output_types = (
            tf.float32,
            tf.float32,
            tf.bool,
            tf.int32,
            tf.float32,
            tf.bool,
        )
        self.shape = (
            tf.TensorShape([self.trajectory_size, self.max_num_ped, 2]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, 2]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, self.max_num_ped]),
            tf.TensorShape([]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, 2]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, self.max_num_ped]),
        )
