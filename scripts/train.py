#!/usr/bin/env python

import os
import time
import logging
import argparse
import tensorflow as tf

import utils
from logger import setLogger
from model import SocialModel


PHASE = "TRAIN"


def main():
    # Parse the arguments received from command line
    parser = argparse.ArgumentParser(description="Train a social LSTM")
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
        type=str,
        metavar="path",
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

    for experiment in experiments:
        # Load the parameters
        hparams = utils.YParams(experiment)
        # Define the logger
        setLogger(hparams, args, PHASE)

        remainSpaces = 29 - len(hparams.name)
        logging.info(
            "\n"
            + "--------------------------------------------------------------------------------\n"
            + "|                            Training experiment: "
            + hparams.name
            + " " * remainSpaces
            + "|\n"
            + "--------------------------------------------------------------------------------\n"
        )

        trajectory_size = hparams.obsLen + hparams.predLen

        logging.info("Loading the training datasets...")
        train_loader = utils.DataLoader(
            hparams.dataPath,
            hparams.trainDatasets,
            hparams.trainMaps,
            hparams.semanticMaps,
            hparams.trainMapping,
            hparams.homography,
            num_labels=hparams.numLabels,
            delimiter=hparams.delimiter,
            skip=hparams.skip,
            max_num_ped=hparams.maxNumPed,
            trajectory_size=trajectory_size,
            neighborood_size=hparams.neighborhoodSize,
        )
        logging.info("Loading the validation datasets...")
        val_loader = utils.DataLoader(
            hparams.dataPath,
            hparams.validationDatasets,
            hparams.validationMaps,
            hparams.semanticMaps,
            hparams.validationMapping,
            hparams.homography,
            num_labels=hparams.numLabels,
            delimiter=hparams.delimiter,
            skip=hparams.skip,
            max_num_ped=hparams.maxNumPed,
            trajectory_size=trajectory_size,
            neighborood_size=hparams.neighborhoodSize,
        )

        logging.info("Creating the training and validation dataset pipeline...")
        dataset = utils.TrajectoriesDataset(
            train_loader,
            val_loader=val_loader,
            batch=False,
            shuffle=hparams.shuffle,
            prefetch_size=hparams.prefetchSize,
        )

        hparams.add_hparam("learningRateSteps", train_loader.num_sequences)

        logging.info("Creating the model...")
        start = time.time()
        model = SocialModel(dataset, hparams, phase=PHASE)
        end = time.time() - start
        logging.debug("Model created in {:.2f}s".format(end))

        # Define the path to where save the model and the checkpoints
        if hparams.modelFolder:
            save_model = True
            model_folder = os.path.join(hparams.modelFolder, hparams.name)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
                os.makedirs(os.path.join(model_folder, "checkpoints"))
            model_path = os.path.join(model_folder, hparams.name)
            checkpoints_path = os.path.join(model_folder, "checkpoints", hparams.name)
            # Create the saver
            saver = tf.train.Saver()

        # Zero padding
        padding = len(str(train_loader.num_sequences))

        # ============================ START TRAINING ============================

        with tf.Session() as sess:
            logging.info(
                "\n"
                + "--------------------------------------------------------------------------------\n"
                + "|                                Start training                                |\n"
                + "--------------------------------------------------------------------------------\n"
            )
            # Initialize all the variables in the graph
            sess.run(tf.global_variables_initializer())

            for epoch in range(hparams.epochs):
                logging.info("Starting epoch {}".format(epoch + 1))

                # ==================== TRAINING PHASE ====================

                # Initialize the iterator of the training dataset
                sess.run(dataset.init_train)

                for sequence in range(train_loader.num_sequences):
                    start = time.time()
                    loss, _ = sess.run([model.loss, model.train_optimizer])
                    end = time.time() - start

                    logging.info(
                        "{:{width}d}/{} epoch: {} time/Batch = {:.2f}s. Loss = {:.4f}".format(
                            sequence + 1,
                            train_loader.num_sequences,
                            epoch + 1,
                            end,
                            loss,
                            width=padding,
                        )
                    )

                # ==================== VALIDATION PHASE ====================

                logging.info(" ========== Validation ==========")
                # Initialize the iterator of the validation dataset
                sess.run(dataset.init_val)
                loss_val = 0

                for _ in range(val_loader.num_sequences):
                    loss = sess.run(model.loss)
                    loss_val += loss

                mean_val = loss_val / val_loader.num_sequences

                logging.info(
                    "Epoch: {}. Validation loss = {:.4f}".format(epoch + 1, mean_val)
                )

                # Save the model
                if save_model:
                    logging.info("Saving model...")
                    saver.save(
                        sess,
                        checkpoints_path,
                        global_step=epoch + 1,
                        write_meta_graph=False,
                    )
                    logging.info("Model saved...")
            # Save the final model
            if save_model:
                saver.save(sess, model_path)
        tf.reset_default_graph()


if __name__ == "__main__":
    main()
