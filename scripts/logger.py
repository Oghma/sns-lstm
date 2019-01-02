import os
import logging


def setLogger(hparams, args, phase):
    log_file = hparams.name + "-" + phase.lower() + ".log"
    log_folder = None
    level = "INFO"
    formatter = logging.Formatter(
        "[%(asctime)s %(filename)s] %(levelname)s: %(message)s"
    )

    # Check if you have to add a FileHandler
    if args.logFolder is not None:
        log_folder = args.logFolder
    elif hparams.logFolder is not None:
        log_folder = hparams.logFolder

    if log_folder is not None:
        log_file = os.path.join(log_folder, log_file)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

    # Set the level
    if args.logLevel is not None:
        level = args.logLevel.upper()
    elif hparams.logLevel is not None:
        level = hparams.logLevel.upper()

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
