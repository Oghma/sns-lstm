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
from pooling_modules import SocialPooling


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
            batch_size=data["batchSize"],
        )
        logging.info("Loading the validation datasets...")
        val_loader = utils.DataLoader(
            data["dataPath"],
            data["validationDatasets"],
            delimiter=data["delimiter"],
            skip=data["skip"],
            max_num_ped=data["maxNumPed"],
            trajectory_size=trajectory_size,
            batch_size=data["batchSize"],
        )

        logging.info("Creating the training and validation dataset pipeline...")
        dataset = utils.TrajectoriesDataset(
            train_loader,
            val_loader=val_loader,
            batch=False,
            batch_size=data["batchSize"],
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
            dropout=data["dropout"],
        )
        end = time.time() - start
        logging.debug("Model created in {:.2f}s".format(end))

        # Check if there are sequences outside the batches
        train_last_batch = train_loader.num_sequences % train_loader.num_batches
        val_last_batch = val_loader.num_sequences % val_loader.num_batches
        train_num_batches = (
            train_loader.num_batches + 1
            if train_last_batch > 0
            else train_loader.num_batches
        )
        val_num_batches = (
            val_loader.num_batches + 1 if val_last_batch > 0 else val_loader.num_batches
        )

        # Print some info
        logging.info(
            "Number of batches per epoch in training set: {}".format(train_num_batches)
        )
        logging.info(
            "Number of batches per epoch in validation set: {}".format(val_num_batches)
        )

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

                for batch in range(train_loader.num_batches):
                    start = time.time()
                    loss_batch = 0
                    for _ in range(train_loader.batch_size):
                        loss, _ = sess.run([model.loss, model.trainOp])
                        loss_batch += loss
                    end = time.time() - start
                    mean_batch = loss_batch / train_loader.batch_size

                    logging.info(
                        "{}/{} epoch: {} time/Batch = {:.2f}s Total loss = {:.4f} Mean loss = {:.4f}".format(
                            batch + 1,
                            train_num_batches,
                            epoch + 1,
                            end,
                            loss_batch,
                            mean_batch,
                        )
                    )

                if train_last_batch > 0:
                    loss_batch = 0
                    for _ in range(train_last_batch):
                        loss, _ = sess.run([model.loss, model.trainOp])
                        loss_batch += loss
                    mean_batch = loss_batch / train_last_batch

                    logging.info(
                        "{}/{} epoch: {} time/Batch = {:.2f}s Total loss = {:.4f} Mean = {:.4f}".format(
                            train_num_batches,
                            train_num_batches,
                            epoch + 1,
                            end,
                            loss_batch,
                            mean_batch,
                        )
                    )

                # ==================== VALIDATION PHASE ====================

                logging.info(" ========== Validation ==========")
                # Initialize the iterator of the validation dataset
                sess.run(dataset.init_val)
                loss_val = 0

                for batch in range(val_loader.num_batches):
                    for _ in range(val_loader.batch_size):
                        loss = sess.run(model.loss)
                        loss_val += loss

                for _ in range(val_last_batch):
                    loss = sess.run(model.loss)
                    loss_val += loss

                mean_val = loss_val / (val_loader.batch_size * val_num_batches)

                logging.info(
                    "Epoch: {} Validation loss = {:.4f} Mean = {:.4f}".format(
                        epoch + 1, loss_val, mean_val
                    )
                )

                # Save the model
                if save_model:
                    logging.info("Saving model...")
                    saver.save(
                        sess,
                        checkpoints_path,
                        global_step=epoch,
                        write_meta_graph=False,
                    )
                    logging.info("Model saved...")
            # Save the final model
            if save_model:
                saver.save(sess, model_path)
        tf.reset_default_graph()


if __name__ == "__main__":
    main()
