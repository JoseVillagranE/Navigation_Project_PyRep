import os
import time
import logging


def get_logger(save_path, log_file):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(save_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger



if __name__ == "__main__":

    logger = get_logger("./loggers", "Planning_Info.log")
    logger.info("Not Found Path")
