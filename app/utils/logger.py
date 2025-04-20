import logging
import os

from dotenv import load_dotenv

load_dotenv()
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
if ENVIRONMENT == "development":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
