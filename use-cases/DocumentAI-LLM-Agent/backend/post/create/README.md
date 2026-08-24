# Chapter 05: Creating Sales Orders

## Overview
This chapter teaches you how to create new Sales Orders in S/4HANA using POST operations and the deep-insert pattern. You'll learn to handle CSRF tokens and build proper payloads.

## Learning Objectives

By the end of this chapter, you will be able to:
- Fetch and use CSRF tokens for write operations
- Read a template Sales Order to understand structure
- Build a Sales Order payload with header and items
- Create a Sales Order using POST deep-insert
- Handle creation errors and validation messages

## Prerequisites

- Completed Chapter 04 (Reading Sales Orders)
- Understanding of OData $expand
- Access to an S/4HANA system with create authorization

## Key Concepts

### CSRF Token
Cross-Site Request Forgery token is required for all write operations (POST, PATCH, DELETE) in S/4HANA. The token must be fetched before the write operation and passed along with cookies.

### Deep-Insert
OData deep-insert allows creating a header entity and its related entities (items, partners) in a single POST request.

### Template Approach
The safest way to create a Sales Order is to read an existing order first and use it as a template to ensure all required fields are included.

## Creating Sales Orders

### Complete Workflow Overview

The process of creating a Sales Order in S/4HANA follows a specific sequence that ensures security and data integrity:

1. **Security First:** We fetch a CSRF token to authenticate our write operation
2. **Template Approach:** We read an existing Sales Order to understand the required structure
3. **Data Preparation:** We build a new payload based on the template, ensuring all required fields are present
4. **Creation:** We send the payload to S/4HANA and handle the response

This workflow minimizes errors by using proven data structures and follows S/4HANA's security requirements.

### Step 1: Fetch CSRF Token

**What we're doing:** Before we can create, update, or delete any data in S/4HANA, we need to obtain a CSRF (Cross-Site Request Forgery) token. This is a security mechanism that prevents unauthorized write operations.

**Why this is necessary:** S/4HANA requires a CSRF token for all write operations (POST, PATCH, DELETE) to ensure that requests are legitimate and not coming from malicious sources.

**How it works:** We make a GET request with the special header `X-CSRF-Token: Fetch` to tell the server we want a token. The server responds with the token in the response headers, along with cookies that must be used together with the token.

```python
def fetch_csrf(session: requests.Session) -> tuple[str, dict]:
    """
    Fetch CSRF token required for POST/PATCH/DELETE operations.
    
    Returns:
        Tuple of (token_string, cookies_dict)
    """
    # Try GET with $top=0 (fastest method)
    # We use $top=0 to get no data, just the token - this is the fastest approach
    h = {"X-CSRF-Token": "Fetch", "Accept": "application/json"}
    p = {"sap-client": CLIENT, "$top": "0"}
    r = session.get(f"{API_SO}/A_SalesOrder", params=p, headers=h, timeout=60)
    token = r.headers.get("X-CSRF-Token")
    if token:
        return token, r.cookies

    # Fallback: GET $metadata
    # If the first method fails, we try getting the metadata endpoint
    h = {"X-CSRF-Token": "Fetch", "Accept": "application/xml"}
    r = session.get(f"{API_SO}/$metadata", 
                    params={"sap-client": CLIENT}, 
                    headers=h, 
                    timeout=60)
    token = r.headers.get("X-CSRF-Token")
    if token:
        return token, r.cookies

    raise RuntimeError(f"CSRF token fetch failed: status={r.status_code}")
```

### Step 2: Read Template Sales Order

**What we're doing:** We're reading an existing Sales Order to use as a template for creating a new one. This ensures we have all the required organizational data and field structures.

**Why this approach:** Instead of guessing what fields are required, we copy the structure from an existing order. This guarantees we have valid organizational units, customer data, and material information.

**Key details:**
- We use `$expand` to get both header data and related items/partners
- We pad the Sales Order number with zeros to match S/4HANA's 10-digit format
- We URL-encode the Sales Order key for the API call

```python
def get_template_so(session: requests.Session, so_number: str) -> dict:
    """Read an existing Sales Order to use as a template."""
    # Pad the Sales Order number with leading zeros to 10 digits
    # This matches S/4HANA's internal format for Sales Order numbers
    so_key = quote(so_number.zfill(10), safe="")
    url = f"{API_SO}/A_SalesOrder('{so_key}')"
    
    params = {
        "sap-client": CLIENT,
        "$format": "json",
        # Expand to get both header and related items/partners in one call
        "$expand": "to_Item,to_Partner"
    }
    
    r = session.get(url, params=params, timeout=120)
    r.raise_for_status()
    
    data = r.json()
    return (data.get("d") or data)
```

### Step 3: Build Sales Order Payload

**What we're doing:** We're constructing the JSON payload that will be sent to S/4HANA to create a new Sales Order. We extract the necessary data from our template and build a minimal but complete structure.

**Why we need this:** S/4HANA requires specific organizational data and at least one item to create a Sales Order. We copy the organizational structure from the template to ensure all required fields are present.

**Key components:**
- **Header data:** Sales organization, distribution channel, customer, currency, dates
- **Item data:** Material, quantity, plant, shipping point, pricing information
- **Deep-insert structure:** We use OData's deep-insert pattern to create header and items in one request

```python
def build_minimal_payload(src: dict) -> dict:
    """
    Build a minimal Sales Order payload from a template.
    
    Args:
        src: Template sales order
    
    Returns:
        Payload dictionary for deep-insert
    """
    # Header data - copy organizational structure from template
    # These fields are required for any Sales Order
    hdr = {
        "SalesOrderType":        src.get("SalesOrderType") or "OR",
        "SalesOrganization":     src.get("SalesOrganization"),
        "DistributionChannel":   src.get("DistributionChannel"),
        "OrganizationDivision":  src.get("OrganizationDivision"),
        "SoldToParty":           src.get("SoldToParty"),
        "TransactionCurrency":   src.get("TransactionCurrency") or "EUR",
        "PricingDate":           sap_date(dt.date.today()),
        "RequestedDeliveryDate": sap_date(dt.date.today() + dt.timedelta(days=7)),
    }

    # At least one item is required for a Sales Order
    # We get the first item from the template to copy its structure
    items = (src.get("to_Item") or {}).get("results", [])
    if not items:
        raise ValueError("Template has no items")
    
    it = items[0]
    
    # Build new item based on template item
    # We copy material, quantity, and organizational data
    new_item = {
        "Material":                it.get("Material"),
        "RequestedQuantity":       it.get("RequestedQuantity") or "1",
        "RequestedQuantityUnit":   it.get("RequestedQuantityUnit") or "PC",
        "ProductionPlant":         it.get("ProductionPlant") or "C000",
        "ShippingPoint":           it.get("ShippingPoint") or "VS01",
        "PricingDate":             sap_date(dt.date.today()),
        "IncotermsClassification": it.get("IncotermsClassification") or "EXW",
        "IncotermsLocation1":      it.get("IncotermsLocation1") or "ex works",
    }

    # Deep-insert structure: header + items in one payload
    # This allows us to create the Sales Order and its items in a single POST request
    payload = hdr
    payload["to_Item"] = {"results": [new_item]}

    return payload
```

### Step 4: Create Sales Order

**What we're doing:** We're sending a POST request to S/4HANA to create the new Sales Order using the payload we built. This is the actual creation step that will result in a new Sales Order number being generated.

**Why this step is critical:** This is where all our preparation pays off. We use the CSRF token for security, send our payload with proper headers, and handle any errors that might occur during creation.

**Key requirements:**
- **CSRF token:** Must be included in headers and cookies for security
- **Proper headers:** Content-Type, Accept, and special OData headers
- **Error handling:** S/4HANA provides detailed error messages that help diagnose issues
- **Response handling:** We get back the created Sales Order with its new number

```python
def create_sales_order(session: requests.Session, payload: dict) -> dict:
    """
    Create a new Sales Order via POST deep-insert.
    
    Args:
        session: Authenticated session
        payload: Sales order data
    
    Returns:
        Created sales order data from server
    """
    # Get CSRF token - required for all write operations
    token, cookies = fetch_csrf(session)

    url = f"{API_SO}/A_SalesOrder"
    params = {"sap-client": CLIENT}
    
    # Headers required for S/4HANA POST operations
    headers = {
        "X-CSRF-Token": token,                    # Security token
        "X-Requested-With": "XMLHttpRequest",     # AJAX request indicator
        "Prefer": "return=representation",        # Return created data
        "Accept": "application/json",             # Expected response format
        "Content-Type": "application/json",      # Payload format
    }

    # Send POST request with our payload
    r = session.post(
        url, 
        params=params, 
        headers=headers, 
        cookies=cookies,                          # CSRF cookies from token fetch
        data=json.dumps(payload),                 # Our Sales Order data
        timeout=120
    )
    
    # Handle any HTTP errors with detailed error reporting
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"\nCREATE FAILED: HTTP {e.response.status_code}")
        try:
            # Try to parse error details from S/4HANA
            error_data = r.json()
            print("Error details:", json.dumps(error_data, indent=2))
        except:
            # If JSON parsing fails, show raw response
            print("Response:", r.text[:500])
        raise

    # Parse successful response
    data = r.json()
    return (data.get("d") or data)
```

## Expected Output

```
=== TEMPLATE (summary) ===
{
  "SalesOrder": "1",
  "SalesOrderType": "OR",
  "SalesOrganization": "1090",
  "DistributionChannel": "92",
  "SoldToParty": "M_CR10D101",
  "TransactionCurrency": "EUR"
}

=== PAYLOAD (preview) ===
{
  "SalesOrderType": "OR",
  "SalesOrganization": "1090",
  ...
  "to_Item": {
    "results": [
      {
        "Material": "M_CR_1001",
        "RequestedQuantity": "1",
        ...
      }
    ]
  }
}

=== SERVER RESPONSE ===
{
  "SalesOrder": "1420",
  "SalesOrderType": "OR",
  ...
}

Created Sales Order: 1420
```

## Common Creation Issues

### Issue 1: CSRF Token Missing (403 Forbidden)
**What happens:** S/4HANA rejects the request with HTTP 403 because no CSRF token was provided.

**Why it occurs:** S/4HANA requires CSRF tokens for all write operations as a security measure to prevent cross-site request forgery attacks.

**Solution:** Always fetch CSRF token before POST operations using the `fetch_csrf()` function and include both the token in headers and cookies in the request.

### Issue 2: Missing Required Fields (400 Bad Request)
**What happens:** S/4HANA returns HTTP 400 with detailed error messages about missing or invalid fields.

**Why it occurs:** Sales Orders require specific organizational data (Sales Organization, Distribution Channel, etc.) that must be valid for the system configuration.

**Solution:** Use template approach to copy all organizational data from an existing Sales Order. This ensures all required fields are present and valid.

### Issue 3: Material Not Found (400 Bad Request)
**What happens:** The system cannot find the specified material in the Sales Order item.

**Why it occurs:** The material either doesn't exist, isn't assigned to the Sales Organization, or isn't valid for the Distribution Channel.

**Solution:** Verify material exists and is assigned to the sales organization. Check that the material is valid for the specific Sales Organization/Distribution Channel combination.

### Issue 4: Customer Not Valid (400 Bad Request)
**What happens:** The SoldToParty (customer) is not valid for the specified Sales Organization.

**Why it occurs:** Customers must be properly assigned to Sales Organizations in S/4HANA's master data.

**Solution:** Ensure SoldToParty exists and is valid for the sales organization. Check customer master data assignments in S/4HANA.

### Issue 5: Date Format Issues (400 Bad Request)
**What happens:** Date fields are rejected due to incorrect format.

**Why it occurs:** S/4HANA expects dates in specific formats (usually ISO format or SAP internal format).

**Solution:** Use the `sap_date()` helper function to convert Python dates to the correct SAP format.

## Key Takeaways

1. **Security is mandatory:** Always fetch CSRF token before write operations - S/4HANA will reject any POST/PATCH/DELETE request without proper authentication.

2. **Token and cookies work together:** Pass token in headers AND cookies from the same request - both are required for S/4HANA to validate the request.

3. **Template approach prevents errors:** Use template approach for complex entities - copying from existing data ensures all required fields are present and valid.

4. **Deep-insert is powerful:** Deep-insert creates header + items in one request - this is more efficient than creating header first, then items separately.

5. **URL parameters matter:** Do NOT include `$format` in POST URL parameters - this can cause issues with the request processing.

6. **Error handling is crucial:** S/4HANA provides detailed error messages - always check the response for specific field validation errors.

7. **Organizational data is key:** Sales Orders require valid organizational units - ensure Sales Organization, Distribution Channel, and Division are properly configured.

8. **Master data relationships:** Materials and customers must be assigned to the correct organizational units - check master data assignments before creating orders.

## Next Steps

Proceed to **Chapter 06: Updating Data** to learn PATCH operations for modifying existing Sales Orders.

## Files in This Chapter

- `README.md` - This file
- `create_sales_order.py` - Complete example script

