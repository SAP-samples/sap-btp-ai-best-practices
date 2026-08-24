"""
Shared configuration helper for S/4HANA integration
Load credentials from .env file in the project root
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory
parent_dir = Path(__file__).parent.parent
load_dotenv(parent_dir / ".env")

# System configuration
BASE_URL = os.getenv("S4_BASE_URL", "").rstrip("/")
CLIENT = os.getenv("S4_CLIENT", "550")
USER = os.getenv("S4_USERNAME")
PWD = os.getenv("S4_PASSWORD")

# API endpoints
API_BP = f"{BASE_URL}/sap/opu/odata/sap/API_BUSINESS_PARTNER"
API_SO = f"{BASE_URL}/sap/opu/odata/sap/API_SALES_ORDER_SRV"

# SSL/TLS settings
VERIFY = os.getenv("S4_VERIFY", "false").lower() not in ("0", "false", "no")
CA_BUNDLE = os.getenv("S4_CA_BUNDLE")
VERIFY = CA_BUNDLE if CA_BUNDLE else VERIFY

# Suppress SSL warnings if verification is disabled
if not VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Validate configuration
if not all([BASE_URL, USER, PWD]):
    print("ERROR: Missing credentials!")
    print("Please ensure S4_BASE_URL, S4_USERNAME, and S4_PASSWORD are set in .env file")
    sys.exit(1)

