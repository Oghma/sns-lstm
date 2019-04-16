#!/usr/bin/env python

import os
import time
import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf
from beautifultable import BeautifulTable

import utils
from logger import setLogger
from model import SocialModel


PHASE = "SAMPLE"


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
        "-l",
        "--logLevel",
        help="logging level of the logger. Default is INFO",
        metavar="level",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--logFolder",
        help="path to the folder where to save the logs. If None, logs are only printed in stderr",
        metavar="path",
        type=str,
    )
    parser.add_argument(
        "-ns",
        "--noSaveCoordinates",
        help="Flag to not save the predicted and ground truth coordinates",
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
        hparams = utils.YParams(experiment)
        # Define the logger
        setLogger(hparams, args, PHASE)

        remainSpaces = 29 - len(hparams.name)
        logging.info(
            "\n"
            + "--------------------------------------------------------------------------------\n"
            + "|                            Sampling experiment: "
            + hparams.name
            + " " * remainSpaces
            + "|\n"
            + "--------------------------------------------------------------------------------\n"
        )

        trajectory_size = hparams.obsLen + hparams.predLen

        saveCoordinates = False
        if args.noSaveCoordinates is True:
            saveCoordinates = False
        elif hparams.saveCoordinates:
            saveCoordinates = hparams.saveCoordinates

        if saveCoordinates:
            coordinates_path = os.path.join("coordinates", hparams.name)
            if not os.path.exists("coordinates"):
                os.makedirs("coordinates")

        logging.info("Loading the test datasets...")
        test_loader = utils.DataLoader(
            hparams.dataPath,
            hparams.testDatasets,
            hparams.testMaps,
            hparams.semanticMaps,
            hparams.testMapping,
            hparams.homography,
            num_labels=hparams.numLabels,
            delimiter=hparams.delimiter,
            skip=hparams.skip,
            max_num_ped=hparams.maxNumPed,
            trajectory_size=trajectory_size,
            neighborood_size=hparams.neighborhoodSize,
        )

        logging.info("Creating the test dataset pipeline...")
        dataset = utils.TrajectoriesDataset(
            test_loader,
            val_loader=None,
            batch=False,
            shuffle=hparams.shuffle,
            prefetch_size=hparams.prefetchSize,
        )

        logging.info("Creating the model...")
        start = time.time()
        model = SocialModel(dataset, hparams, phase=PHASE)
        end = time.time() - start
        logging.debug("Model created in {:.2f}s".format(end))

        # Define the path to the file that contains the variables of the model
        model_folder = os.path.join(hparams.modelFolder, hparams.name)
        model_path = os.path.join(model_folder, hparams.name)

        # Create a saver
        saver = tf.train.Saver()

        # Add to the computation graph the evaluation functions
        ade_sequence = utils.average_displacement_error(
            model.new_pedestrians_coordinates[-hparams.predLen :],
            model.pedestrians_coordinates[-hparams.predLen :],
            model.num_peds_frame,
        )

        fde_sequence = utils.final_displacement_error(
            model.new_pedestrians_coordinates[-1],
            model.pedestrians_coordinates[-1],
            model.num_peds_frame,
        )

        ade = 0
        fde = 0
        coordinates_predicted = []
        coordinates_gt = []
        peds_in_sequence = []

        # Zero padding
        padding = len(str(test_loader.num_sequences))

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
                    "Sample trajectory number {:{width}d}/{}".format(
                        seq + 1, test_loader.num_sequences, width=padding
                    )
                )

                ade_value, fde_value, coordinates_pred_value, coordinates_gt_value, num_peds = sess.run(
                    [
                        ade_sequence,
                        fde_sequence,
                        model.new_pedestrians_coordinates,
                        model.pedestrians_coordinates,
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
            results.append_row([hparams.name, ade, fde])

            if saveCoordinates:
                coordinates_predicted = np.array(coordinates_predicted)
                coordinates_gt = np.array(coordinates_gt)
                saveCoords(
                    coordinates_predicted,
                    coordinates_gt,
                    peds_in_sequence,
                    hparams.predLen,
                    coordinates_path,
                )
        tf.reset_default_graph()
    logging.info("\n{}".format(results))


def saveCoords(pred, coordinates_gt, peds_in_sequence, pred_len, coordinates_path):
    """Save a pickle file with a dictionary containing the predicted coordinates,
    the ground truth coordiantes and the number of pedestrian in sequence.

    Args:
      pred: numpy array [num_sequence, trajectory_size, max_num_ped, 2]. The
        predicted coordinates.
      coordinates_gt: numpy array [num_sequence, max_num_ped, trajectory_size,
        2]. The ground truth coordinates.
      peds_in_sequence: numpy array [num_sequence]. The number of pedestrian in
        each sequence.
      pred_len: int. Number of prediction time-steps.
      coordinates_path: string. Path to where to save the coordinates.

    """
    coordinates = {"groundTruth": coordinates_gt, "pedsInSequence": peds_in_sequence}
    coordinates_pred = coordinates_gt.copy()

    for index, sequence in enumerate(pred):
        coords = sequence[-pred_len:, : peds_in_sequence[index]]
        coordinates_pred[index, -pred_len:, : peds_in_sequence[index]] = coords

    coordinates["predicted"] = coordinates_pred
    with open(coordinates_path + "pkl", "wb") as fp:
        pickle.dump(coordinates, fp)


if __name__ == "__main__":
    main()
