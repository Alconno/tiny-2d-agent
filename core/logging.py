import logging
import sys

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def get_logger(name: str):
    return logging.getLogger(name)
