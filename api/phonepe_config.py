import os
from dotenv import load_dotenv

load_dotenv()

PHONEPE_BASE_URL = os.getenv("PHONEPE_BASE_URL")
MERCHANT_ID = os.getenv("PHONEPE_MERCHANT_ID")
SALT_KEY = os.getenv("PHONEPE_SALT_KEY")
SALT_INDEX = os.getenv("PHONEPE_SALT_INDEX")
REDIRECT_URL = os.getenv("PHONEPE_REDIRECT_URL")
