"""Data loader classes for the LSTMs models. The classes load datasets,
preprocess them and create generators that return sequences of trajectories or
batches of trajectories.

"""
import os
import random
import logging
import numpy as np
import tensorflow as tf


class DataLoader:
    """Data loader class that load the given datasets, preprocess them and create
    two generators that return sequences of trajectories or batches of
    trajectories.

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
            Generator object which contains a list containing sequences of
            trajectories of size batch_size and a list containing the number of
            pedestrian in the sequences.

        """
        it = self.next_sequence()
        for batch in range(self.num_batches):
            batch = []
            peds_in_batch = []

            for size in range(self.batch_size):
                data = next(it)
                batch.append(data[0])
                peds_in_batch.append(data[0])
            yield batch, peds_in_batch

    def next_sequence(self):
        """Generator method that returns an iterator pointing to the next sequence.

        Returns:
            Generator object which contains a sequence of trajectories and the
            number of pedestrian in the trajectory.

        """
        # Iterate through all sequences
        for dataset in self.__trajectories:
            # Every dataset
            for sequence in dataset:
                yield self.__get_sequence(sequence)

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
        the trajectories with length trajectory_size.

        """
        # Keep only the trajectories trajectory_size long
        self.__trajectories = []
        self.num_sequences = 0

        for dataset in self.__frames:
            # Initialize the array of trajectories for the current dataset.
            trajectories = []
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
                    frames = sequence[sequence[:, 1] == ped, :]
                    if frames.shape[0] == self.trajectory_size:
                        traj_frame.append(frames)
                # If no trajectory is long enough traj_frame is empty
                if traj_frame:
                    trajectories.append(traj_frame)
                    self.num_sequences += 1
                # If skip is True, update the index with a random value
                if self.skip is True:
                    i += random.randint(0, self.trajectory_size)
                else:
                    i += self.skip
            self.__trajectories.append(trajectories)

        # num_batches counts only full batches. It discards the remaining
        # sequences
        self.num_batches = int(self.num_sequences / self.batch_size)
        logging.info("There are {} sequences in loader".format(self.num_sequences))
        logging.info("There are {} batches in loader".format(self.num_batches))

    def __get_sequence(self, trajectories):
        """Returns a sequence of trajectories of shape [maxNumPed, trajectory_size, 2]

        Args:
          trajectories: list of numpy array. Each array is a trajectory

        Returns:
          tuple containing a numpy array with shape [max_num_ped,
          trajectory_size, 2] that contains all the trajectories and the number
          of pedestrian in the sequence

        """
        sequence = np.zeros((self.max_num_ped, self.trajectory_size, 2))
        peds_in_sequence = len(trajectories)

        for index, trajectory in enumerate(trajectories):
            sequence[index] = trajectory[:, [2, 3]]
        return sequence, peds_in_sequence

    def __type_and_shape(self):
        """Define the type and the shape of the arrays that tensorflow will use"""
        self.output_types = (tf.float32, tf.int32)
        self.shape = (
            tf.TensorShape([self.max_num_ped, self.trajectory_size, 2]),
            tf.TensorShape([]),
        )
