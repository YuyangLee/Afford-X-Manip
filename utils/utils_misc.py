import os
from datetime import datetime
from loguru import logger


def prep_logdir():
    time_tag = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    logdir = os.path.join("outputs", time_tag)
    os.makedirs(logdir, exist_ok=True)
    logger.add(os.path.join(logdir, "run_tom.log"))
    logger.info(f"Results directory: [{logdir}], logging to [run_tom.log]")
    return logdir
