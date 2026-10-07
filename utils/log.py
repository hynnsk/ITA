import logging


def set_logger(log_file, level):
    """set logger"""
    logger = logging.getLogger("logger")
    logger.setLevel(level)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    if log_file is not None:
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)

    return logger
