#!/usr/bin/env python

import os
import time
import yaml
import logging
import argparse
import numpy as np
import tensorflow as tf

import utils
from model import SocialModel
from coordinates_helpers import sample_helper
from losses import social_loss_function
from position_estimates import social_sample_position_estimate
from pooling_modules import SocialPooling
from beautifultable import BeautifulTable


def logger(data, args):
    log_file = data["name"] + "-sample.log"
    log_folder = None
    level = "INFO"
    formatter = logging.Formatter(
        "[%(asctime)s %(filename)s] %(levelname)s: %(message)s"
    )

    # Check if you have to add a FileHandler
    if args.logFolder is not None:
        log_folder = args.logFolder
    elif "logFolder" in data:
        log_folder = data["logFolder"]

    if log_folder is not None:
        log_file = os.path.join(log_folder, log_file)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

    # Set the level
    if args.logLevel is not None:
        level = args.logLevel.upper()
    elif "logLevel" in data:
        level = data["logLevel"].upper()

    # Get the logger
    logger = logging.getLogger()
    # Remove handlers added previously
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logger.removeHandler(handler)
    if log_folder is not None:
        # Add a FileHandler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Add a StreamHandler that display on sys.stderr
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Set the level
    logger.setLevel(level)


def main():
    parser = argparse.ArgumentParser(
        description="Sample new trajectories with a social LSTM"
    )
    parser.add_argument(
        "modelParams",
        type=str,
        help="Path to the file or folder with the parameters of the experiments",
    )
    parser.add_argument(
        "-logL",
        "--logLevel",
        help="logging level of the logger. Default is INFO",
        metavar="level",
        type=str,
    )
    parser.add_argument(
        "-logF",
        "--logFolder",
        help="path to the folder where to save the logs. If None, logs are only printed in stderr",
        type=str,
    )
    parser.add_argument(
        "-saveC",
        "--saveCoordinates",
        help="Flag for save the predicted and ground truth coordinates",
        action="store_true",
    )
    args = parser.parse_args()

    if os.path.isdir(args.modelParams):
        names_experiments = os.listdir(args.modelParams)
        experiments = [
            os.path.join(args.modelParams, experiment)
            for experiment in names_experiments
        ]
    else:
        experiments = [args.modelParams]

    # Table will show the metrics of each experiment
    results = BeautifulTable()
    results.column_headers = ["Name experiment", "ADE", "FDE"]

    for experiment in experiments:
        # Load the parameters
        with open(experiment) as fp:
            data = yaml.load(fp)
        # Define the logger
        logger(data, args)

        remainSpaces = 29 - len(data["name"])
        logging.info(
            "\n"
            + "--------------------------------------------------------------------------------\n"
            + "|                            Sampling experiment: "
            + data["name"]
            + " " * remainSpaces
            + "|\n"
            + "--------------------------------------------------------------------------------\n"
        )

        trajectory_size = data["obsLen"] + data["predLen"]
        saveCoordinates = False

        if args.saveCoordinates is not None:
            saveCoordinates = args.saveCoordinates
        elif "saveCoordinates" in data:
            saveCoordinates = data["saveCoordinates"]

        if saveCoordinates:
            coordinates_path = os.path.join("coordinates", data["name"])
            if not os.path.exists("coordinates"):
                os.makedirs("coordinates")

        logging.info("Loading the test datasets...")
        test_loader = utils.DataLoader(
            data["dataPath"],
            data["testDatasets"],
            delimiter=data["delimiter"],
            skip=data["skip"],
            max_num_ped=data["maxNumPed"],
            trajectory_size=trajectory_size,
            batch_size=data["batchSize"],
        )

        logging.info("Creating the test dataset pipeline...")
        dataset = utils.TrajectoriesDataset(
            test_loader,
            val_loader=None,
            batch=False,
            batch_size=data["batchSize"],
            prefetch_size=data["prefetchSize"],
        )

        logging.info("Creating the helper for the coordinates")
        helper = sample_helper(data["obsLen"])

        pooling_module = None
        if data["poolingModule"] == "social":
            logging.info("Creating the {} pooling".format(data["poolingModule"]))
            pooling_class = SocialPooling(
                grid_size=data["gridSize"],
                neighborhood_size=data["neighborhoodSize"],
                max_num_ped=data["maxNumPed"],
                embedding_size=data["embeddingSize"],
                rnn_size=data["lstmSize"],
            )
            pooling_module = pooling_class.pooling

        logging.info("Creating the model...")
        start = time.time()
        model = SocialModel(
            dataset,
            helper,
            social_sample_position_estimate,
            social_loss_function,
            pooling_module,
            lstm_size=data["lstmSize"],
            max_num_ped=data["maxNumPed"],
            trajectory_size=trajectory_size,
            embedding_size=data["embeddingSize"],
            learning_rate=data["learningRate"],
            dropout=data["dropout"],
        )
        end = time.time() - start
        logging.debug("Model created in {:.2f}s".format(end))

        # Define the path to the file that contains the variables of the model
        data["modelFolder"] = os.path.join(data["modelFolder"], data["name"])
        model_path = os.path.join(data["modelFolder"], data["name"])

        # Create a saver
        saver = tf.train.Saver()

        # Add to the computation graph the evaluation functions
        ade_sequence = utils.average_displacement_error(
            model.new_coordinates[-data["predLen"] :],
            model.input_data[:, -data["predLen"] :],
            model.num_peds_frame,
        )

        fde_sequence = utils.final_displacement_error(
            model.new_coordinates[-1], model.input_data[:, -1], model.num_peds_frame
        )

        ade = 0
        fde = 0
        coordinates_predicted = []
        coordinates_gt = []
        peds_in_sequence = []

        # ============================ START SAMPLING ============================

        with tf.Session() as sess:
            # Restore the model trained
            saver.restore(sess, model_path)

            # Initialize the iterator of the sample dataset
            sess.run(dataset.init_train)

            logging.info(
                "\n"
                + "--------------------------------------------------------------------------------\n"
                + "|                                Start sampling                                |\n"
                + "--------------------------------------------------------------------------------\n"
            )

            for seq in range(test_loader.num_sequences):
                logging.info(
                    "Sample trajectory number {}/{}".format(
                        seq + 1, test_loader.num_sequences
                    )
                )

                ade_value, fde_value, coordinates_pred_value, coordinates_gt_value, num_peds = sess.run(
                    [
                        ade_sequence,
                        fde_sequence,
                        model.new_coordinates,
                        model.input_data,
                        model.num_peds_frame,
                    ]
                )
                ade += ade_value
                fde += fde_value
                coordinates_predicted.append(coordinates_pred_value)
                coordinates_gt.append(coordinates_gt_value)
                peds_in_sequence.append(num_peds)

            ade = ade / test_loader.num_sequences
            fde = fde / test_loader.num_sequences
            logging.info("Sampling finished. ADE: {:.4f} FDE: {:.4f}".format(ade, fde))
            results.append_row([data["name"], ade, fde])

            if saveCoordinates:
                coordinates_predicted = np.array(coordinates_predicted)
                coordinates_gt = np.array(coordinates_gt)
                saveCoords(
                    coordinates_predicted,
                    coordinates_gt,
                    peds_in_sequence,
                    data["predLen"],
                    coordinates_path,
                )
        tf.reset_default_graph()
    logging.info("\n{}".format(results))


def saveCoords(pred, coordinates_gt, peds_in_sequence, pred_len, coordinates_path):
    """Save the predicted, the ground truth coordiantes and the number of
    pedestrian in sequence. The files are in numpy format.

    Args:
      pred: numpy array [num_sequence, trajectory_size - 1,
        max_num_ped, 2]. The predicted coordinates.
      coordinates_gt: numpy array [num_sequence, max_num_ped, trajectory_size,
        2]. The ground truth coordinates.
      peds_in_sequence: numpy array [num_sequence]. The number of pedestrian in
        each sequence.
      pred_len: int. Number of prediction time-steps.
      coordinates_path: string. Path to where to save the coordinates.

    """
    coordinates_pred = coordinates_gt.copy()

    for index, sequence in enumerate(pred):
        coords = sequence[-pred_len:, : peds_in_sequence[index]]
        # Change the shape of the array from [trajectory_size, max_num_ped, 2]
        # to [max_num_ped, trajectory_size, 2]
        coords = np.moveaxis(coords, 0, 1)
        coordinates_pred[index, : peds_in_sequence[index], -pred_len:] = coords

    np.save(coordinates_path + "_gt", coordinates_gt)
    np.save(coordinates_path + "_predicted", coordinates_pred)
    np.save(coordinates_path + "_peds", peds_in_sequence)


if __name__ == "__main__":
    main()
