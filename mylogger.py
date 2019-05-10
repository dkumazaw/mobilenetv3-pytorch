"""Defines a logger"""

import logging


def create_logger(filename: str, logger_prefix: str):
    logging.basicConfig(
        filename=filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    logger = logging.getLogger(logger_prefix)
    consoleHandler = logging.StreamHandler()
    logger.addHandler(consoleHandler)

    return logger
