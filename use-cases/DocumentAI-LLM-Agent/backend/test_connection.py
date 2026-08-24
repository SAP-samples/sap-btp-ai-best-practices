"""
Chapter 01: Test S/4HANA Connection
This script verifies that your environment is properly configured
and you can connect to the S/4HANA system.
"""

import os
import sys
import requests
from dotenv import load_dotenv

def test_connection():
    """Test connection to S/4HANA system."""
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    BASE_URL = os.getenv("S4_BASE_URL")
    CLIENT = os.getenv("S4_CLIENT", "550")
    USER = os.getenv("S4_USERNAME")
    PWD = os.getenv("S4_PASSWORD")
    
    # Verify credentials are loaded
    print("Step 1: Checking configuration...")
    if not all([BASE_URL, USER, PWD]):
        print("ERROR: Missing credentials in .env file")
        print("Please ensure S4_BASE_URL, S4_USERNAME, and S4_PASSWORD are set")
        sys.exit(1)
    print("  ✓ Configuration loaded successfully")
    
    # Prepare SSL verification
    VERIFY = os.getenv("S4_VERIFY", "false").lower() not in ("0", "false", "no")
    CA_BUNDLE = os.getenv("S4_CA_BUNDLE")
    if CA_BUNDLE:
        VERIFY = CA_BUNDLE
    
    # Suppress SSL warnings if verification is disabled
    if not VERIFY:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Create session
    print("\nStep 2: Creating session...")
    session = requests.Session()
    session.auth = (USER, PWD)
    session.verify = VERIFY
    session.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    print("  ✓ Session created")
    
    # Test connection to Business Partner API
    print("\nStep 3: Testing connection to Business Partner API...")
    BASE_URL = BASE_URL.rstrip("/")
    API_BP = f"{BASE_URL}/sap/opu/odata/sap/API_BUSINESS_PARTNER"
    #API_BP = f"{BASE_URL}/sap/opu/odata/sap/A_OperationalAcctgDocItemCube"
    
    try:
        # Try to get count of business partners
        params = {
            "sap-client": CLIENT,
            "sap-language": "EN"
        }
        
        response = session.get(
            f"{API_BP}/A_BusinessPartner/$count",
            params=params,
            headers={"Accept": "text/plain"},
            timeout=30
        )
        
        response.raise_for_status()
        bp_count = int(response.text.strip())
        print(f"  ✓ Connected to S4 system")
        print(f"  ✓ Authentication successful")
        print(f"  ✓ Total Business Partners: {bp_count}")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("  ERROR: Authentication failed")
            print("  Please check your username and password in .env")
            sys.exit(1)
        elif e.response.status_code == 404:
            print("  ERROR: API endpoint not found")
            print("  Please check your S4_BASE_URL in .env")
            sys.exit(1)
        else:
            print(f"  ERROR: HTTP {e.response.status_code}")
            print(f"  {e}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        print("  ERROR: Connection timeout")
        print("  The server took too long to respond")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print("  ERROR: Cannot connect to server")
        print(f"  {e}")
        print("  Please check your S4_BASE_URL and network connection")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Unexpected error: {e}")
        sys.exit(1)
    
    # Test Sales Order API
    print("\nStep 4: Testing Sales Order API...")
    API_SO = f"{BASE_URL}/sap/opu/odata/sap/API_SALES_ORDER_SRV"
    
    try:
        response = session.get(
            f"{API_SO}/A_SalesOrder/$count",
            params=params,
            headers={"Accept": "text/plain"},
            timeout=30
        )
        
        response.raise_for_status()
        so_count = int(response.text.strip())
        print(f"  ✓ Sales Order API accessible")
        print(f"  ✓ Total Sales Orders: {so_count}")
        
    except Exception as e:
        print(f"  WARNING: Sales Order API test failed: {e}")
        print("  You may not have authorization to access this API")
    
    # Success message
    print("\n" + "="*50)
    print("SUCCESS! Your environment is ready!")
    print("="*50)
    print("\nYou can now proceed to Chapter 02: Reading Data")
    print("\nConfiguration Summary:")
    print(f"  Server: {BASE_URL}")
    print(f"  Client: {CLIENT}")
    print(f"  User: {USER}")
    print(f"  SSL Verify: {VERIFY}")

if __name__ == "__main__":
    test_connection()

