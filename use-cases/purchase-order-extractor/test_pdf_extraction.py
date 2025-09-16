#!/usr/bin/env python3

import sys
import os
sys.path.append('backend')

from pdf import extract_purchase_order_data

def test_extraction():
    pdf_path = "/Users/joel/Downloads/Excel & PDF - OV OTC AI/10003322.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    print("Testing PDF extraction...")
    print(f"File: {pdf_path}")
    print("-" * 50)
    
    try:
        result = extract_purchase_order_data(pdf_path, "10003322.pdf")
        print("Extraction result:")
        print(result)
        
        if 'error' in result:
            print(f"\nError occurred: {result['error']}")
        else:
            print(f"\nSuccess! Extracted data:")
            print(f"Cliente: {result.get('cliente', 'N/A')}")
            print(f"Fecha: {result.get('fecha', 'N/A')}")
            print(f"Vendor: {result.get('vendor', 'N/A')}")
            print(f"Productos: {len(result.get('productos', []))} items")
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extraction()
