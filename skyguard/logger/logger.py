import logging
import os
from datetime import datetime

# Log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Logs directory
logs_path = os.path.join(
    os.getcwd(),
    "logs"
)

# Create logs directory
os.makedirs(
    logs_path,
    exist_ok=True
)

# Full log file path
LOG_FILE_PATH = os.path.join(
    logs_path,
    LOG_FILE
)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.info("Logging has started")