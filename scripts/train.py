#!/usr/bin/env python

import os
import time
import yaml
import logging
import argparse
import tensorflow as tf

import utils
from model import SocialModel
from coordinates_helpers import train_helper
from losses import social_loss_function
from position_estimates import social_train_position_estimate
from pooling_layers import SocialPooling


def logger(data, args):
    log_file = data["name"] + "-train.log"
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
        with open(experiment) as fp:
            data = yaml.load(fp)
        # Define the logger
        logger(data, args)

        remainSpaces = 29 - len(data["name"])
        logging.info(
            "\n"
            + "--------------------------------------------------------------------------------\n"
            + "|                            Training experiment: "
            + data["name"]
            + " " * remainSpaces
            + "|\n"
            + "--------------------------------------------------------------------------------\n"
        )

        trajectory_size = data["obsLen"] + data["predLen"]

        logging.info("Loading the training datasets...")
        train_loader = utils.DataLoader(
            data["dataPath"],
            data["trainDatasets"],
            delimiter=data["delimiter"],
            skip=data["skip"],
            max_num_ped=data["maxNumPed"],
            trajectory_size=trajectory_size,
        )
        logging.info("Loading the validation datasets...")
        val_loader = utils.DataLoader(
            data["dataPath"],
            data["validationDatasets"],
            delimiter=data["delimiter"],
            skip=data["skip"],
            max_num_ped=data["maxNumPed"],
            trajectory_size=trajectory_size,
        )

        logging.info("Creating the training and validation dataset pipeline...")
        dataset = utils.TrajectoriesDataset(
            train_loader,
            val_loader=val_loader,
            batch=False,
            prefetch_size=data["prefetchSize"],
        )

        logging.info("Creating the helper for the coordinates")
        helper = train_helper

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
            social_train_position_estimate,
            social_loss_function,
            pooling_module,
            lstm_size=data["lstmSize"],
            max_num_ped=data["maxNumPed"],
            trajectory_size=trajectory_size,
            embedding_size=data["embeddingSize"],
            learning_rate=data["learningRate"],
            clip_norm=data["clippingRatio"],
            l2_norm=data["l2Rate"],
            opt_momentum=data["optimizerMomentum"],
            opt_decay=data["optimizerDecay"],
            lr_steps=train_loader.num_sequences,
            lr_decay=data["learningRateDecay"],
        )
        end = time.time() - start
        logging.debug("Model created in {:.2f}s".format(end))

        # Define the path to where save the model and the checkpoints
        if "modelFolder" in data:
            save_model = True
            data["modelFolder"] = os.path.join(data["modelFolder"], data["name"])
            if not os.path.exists(data["modelFolder"]):
                os.makedirs(data["modelFolder"])
                os.makedirs(os.path.join(data["modelFolder"], "checkpoints"))
            model_path = os.path.join(data["modelFolder"], data["name"])
            checkpoints_path = os.path.join(
                data["modelFolder"], "checkpoints", data["name"]
            )
            # Create the saver
            saver = tf.train.Saver()

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

            for epoch in range(data["epochs"]):
                logging.info("Starting epoch {}".format(epoch + 1))

                # ==================== TRAINING PHASE ====================

                # Initialize the iterator of the training dataset
                sess.run(dataset.init_train)

                for sequence in range(train_loader.num_sequences):
                    start = time.time()
                    loss, _ = sess.run([model.loss, model.trainOp])
                    end = time.time() - start

                    logging.info(
                        "{}/{} epoch: {} time/Batch = {:.2f}s. Loss = {:.4f}".format(
                            sequence + 1,
                            train_loader.num_sequences,
                            epoch + 1,
                            end,
                            loss,
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
