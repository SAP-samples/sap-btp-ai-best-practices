# Import DOX API client
from dotenv import load_dotenv
import os
from sap_business_document_processing import DoxApiClient
import json
from utils import display_capabilities
from sap_business_document_processing.document_information_extraction_client.constants import CONTENT_TYPE_PDF

# Load environment variables from .env file
load_dotenv()


# Retrieve environment variables
url = os.getenv('URL')
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
uaa_url = os.getenv('UAADOMAIN')

# Instantiate object used to communicate with Document Information Extraction REST API
api_client = DoxApiClient(url, client_id, client_secret, uaa_url)

# Get the available document types and corresponding extraction fields
capabilities = api_client.get_capabilities()


# # Check which clients exist for this tenant
# api_client.get_clients()
# # Create a new client with the id 'c_00' and name 'Client 00'
# api_client.create_client(client_id='c_00', client_name='Client 00')

# The constants provide supported content types that can be imported, e.g. for PDF, PNG, JPEG or TIFF files as well as the
# CONTENT_TYPE_UNKNOWN that lets the library fetch the content type automatically based on the file's extension
#

# Specify the fields that should be extracted
header_fields = [
         "documentNumber",
         "taxId",
         "purchaseOrderNumber", 
         "shippingAmount",
         "netAmount",
         "senderAddress",
         "senderName",
         "grossAmount",
         "currencyCode",
         "receiverContact",
         "documentDate",
         "taxAmount",
         "taxRate",
         "receiverName",
         "receiverAddress"
    ]
line_item_fields = [
         "description",
         "netAmount",
         "quantity",
         "unitPrice",
         
    ]

# Extract information from invoice document
document_result = api_client.extract_information_from_document(document_path='bills_data/jan_bill_download.pdf', 
                                                               client_id='default', 
                                                               document_type='invoice', 
                                                               mime_type=CONTENT_TYPE_PDF, 
                                                               header_fields=header_fields, 
                                                               line_item_fields=line_item_fields)


# Check the extracted data

json_object = json.dumps(document_result, indent=2)

# Writing to result.json
with open("result.json", "w") as outfile:
    outfile.write(json_object)

