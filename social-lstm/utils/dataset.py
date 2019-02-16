"""Module that defines the classes that provide the input tensors for the
models. Each class defines at least 2 iterators: the iterator for the
sequence/batch and the iterator for the number of pedestrian in the
sequence/batch. Each class store the iterators in the `tensors` variable.

"""
import tensorflow as tf


class TrajectoriesDataset:
    """Class that defines the tensors iterators used as input for the models.

    TrajectoriesDataset defines the following iterators: the sequence/batch
    iterator, the number of pedestrian in the sequence/batch iterator, the mask
    for each frame of the sequence/batch, the sequence/batch iterator with all
    pedestrians in the frame and the mask for each frame of the all pedestrians
    sequence/batch. The iterators have shape and type defined by the
    train_loader class and they are stored inside the variable `tensors`.

    """

    def __init__(
        self,
        train_loader,
        val_loader=None,
        batch=False,
        shuffle=True,
        batch_size=10,
        prefetch_size=1000,
    ):
        """Constructor of the TrajectoriesDataset class.

        Args:
          train_loader: Object. that provides the sequences. The object must
            have the next_sequence method that returns a generator with the
            sequences.
          val_loader: Object that provides the sequences. The object must have
            the next_sequence method that returns a generator with the
            sequences.
          batch: boolean. If True, TrajectoriesDataset returns batch of
            sequences.
          batch_size: int. Size of the batch.
          prefetch_size: int. Number of batch to prefetch.

        """

        # Create the datasets with the tf.data API. The dataset will use the CPU
        with tf.device("/cpu:0"):
            train_dataset = tf.data.Dataset.from_generator(
                train_loader.next_sequence,
                train_loader.output_types,
                train_loader.shape,
            )
            if val_loader is not None:
                val_dataset = tf.data.Dataset.from_generator(
                    val_loader.next_sequence, val_loader.output_types, val_loader.shape
                )

        # If shuffle is True, add the shuffle option to the dataset
        if shuffle:
            train_dataset = train_dataset.shuffle(prefetch_size)
            if val_loader is not None:
                val_dataset = val_dataset.shuffle(prefetch_size)

        # If batch is True, self.tensors will contain a batch of sequences and
        # not a single sequence
        if batch:
            train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        # Prefetch the sequences or batches if batch is True
        train_dataset = train_dataset.prefetch(prefetch_size)

        if val_loader is not None:
            if batch:
                val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
            val_dataset = val_dataset.prefetch(prefetch_size)

        # Create the iterators
        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes
        )
        # Initialize the iterators
        self.init_train = iterator.make_initializer(train_dataset)
        if val_loader is not None:
            self.init_val = iterator.make_initializer(val_dataset)

        # Tensors is a tuple that contains the output of the iterator
        self.tensors = iterator.get_next()
