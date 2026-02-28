import os
import sys
import logging
from datetime import datetime

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
log_dir = "logs"
log_filename = f"{timestamp}.log"
log_filepath = os.path.join(log_dir, log_filename)

logging_str='[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logging = logging.getLogger("multi-modal_logs")