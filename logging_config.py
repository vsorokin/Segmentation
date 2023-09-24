import logging
import sys
from datetime import datetime, timezone
import os


def configure_logging(result_path=None, loglevel="INFO", log_to_stdout=True):
    handlers = []
    if result_path is not None:
        handlers.append(logging.FileHandler(f"{result_path}/debug.log"))
    if log_to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    logger = logging.getLogger()
    logger.setLevel(loglevel)


def get_result_path(output_dir, suffix=""):
    result_path = f"{output_dir}/{datetime.now().astimezone(timezone.utc).strftime('%Y_%m_%d__%H_%M_%S')}{suffix}"
    # Can't use `logging` here, it's not yet configured.
    print(f"RESULT PATH: {result_path}")
    try:
        os.mkdir(result_path)
    except FileExistsError as e:
        raise RuntimeError(f"Result path already exists: {result_path}") from e

    return result_path
