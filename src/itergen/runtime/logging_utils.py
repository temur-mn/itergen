"""
Run logging helpers.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_run_logger(log_dir=None, name="generator"):
    package_dir = Path(__file__).resolve().parents[1]
    default_log_dir = package_dir / "logs"

    if log_dir is None:
        resolved_log_dir = default_log_dir
    else:
        text = str(log_dir).strip()
        resolved_log_dir = Path(text).expanduser() if text else default_log_dir

    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = resolved_log_dir / f"run_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(stream_handler)

    return logger, str(log_path)
