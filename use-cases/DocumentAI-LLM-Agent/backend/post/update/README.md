# Chapter 06: Updating Data (PATCH Operations)

## Overview
This chapter covers updating existing data in S/4HANA using PATCH operations. You'll learn to modify Business Partners, Sales Order headers, and line items with detailed explanations of each step.

## Learning Objectives

By the end of this chapter, you will be able to:
- Update Business Partner data using PATCH
- Modify Sales Order header fields
- Update individual Sales Order line items
- Handle 204 No Content responses
- Verify updates after modification
- Understand the step-by-step process of each update operation

## Prerequisites

- Completed Chapter 05 (Creating Sales Orders)
- Understanding of CSRF tokens

## Key Concepts

### PATCH vs PUT
S/4HANA OData services use PATCH for partial updates (modify only specified fields). PUT would replace the entire entity and is rarely used.

### Response Codes
- **200 OK**: Update successful with response body
- **204 No Content**: Update successful without response body (common in S/4HANA)

### Prefer Header
The `Prefer: return=representation` header requests the server to return the updated entity in the response.

## Step-by-Step Code Explanations

### 1. Session Setup and Authentication

```python
def sess() -> requests.Session:
    """Create an authenticated session with S4 system."""
    s = requests.Session()
    s.auth = (USER, PWD)
    s.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    s.verify = VERIFY
    return s
```

**What this does:**
- Creates a persistent HTTP session for multiple requests
- Sets up basic authentication with username/password
- Configures default headers for JSON communication
- Sets SSL verification based on configuration

**How it works:**
- `requests.Session()` creates a session object that maintains cookies and connection pooling
- `s.auth = (USER, PWD)` sets HTTP Basic Authentication
- Default headers ensure all requests expect and send JSON data
- Session is reused for multiple API calls, improving performance

### 2. CSRF Token Management

```python
def fetch_csrf(session: requests.Session) -> tuple[str, requests.cookies.RequestsCookieJar]:
    """Fetch CSRF token for write operations."""
    h = {"X-CSRF-Token": "Fetch", "Accept": "application/json"}
    p = {"sap-client": CLIENT, "$top": "0"}
    r = session.get(f"{API_SO}/A_SalesOrder", params=p, headers=h, timeout=60)
    token = r.headers.get("X-CSRF-Token")
    if token:
        return token, r.cookies
    raise RuntimeError("CSRF token fetch failed")
```

**What this does:**
- Fetches a CSRF (Cross-Site Request Forgery) token required for write operations
- Uses a lightweight GET request to obtain the token
- Returns both the token and session cookies

**How it works:**
- `"X-CSRF-Token": "Fetch"` header tells S/4HANA to return a CSRF token
- `"$top": "0"` parameter ensures no data is returned, just the token
- The token is extracted from response headers
- Cookies are captured for session management
- This token must be included in all PATCH/POST/DELETE operations

### 3. Business Partner ID Formatting

```python
def bp10(x: str) -> str:
    """Ensure Business Partner ID is 10 digits with leading zeros."""
    return x.zfill(10) if x.isdigit() and len(x) != 10 else x
```

**What this does:**
- Ensures Business Partner IDs are properly formatted for S/4HANA
- Pads numeric IDs with leading zeros to make them 10 digits
- Leaves non-numeric IDs unchanged

**How it works:**
- `x.isdigit()` checks if the string contains only digits
- `len(x) != 10` ensures we only pad if not already 10 digits
- `x.zfill(10)` pads with leading zeros
- Non-numeric IDs (like external IDs) are returned unchanged

### 4. Update Business Partner Function

```python
def update_bp(s: requests.Session, bp_id: str, updates: dict) -> dict:
    """Update a Business Partner using PATCH."""
    token, cookies = fetch_csrf(s)
    
    bid = quote(bp10(bp_id), safe="")
    url = f"{API_BP}/A_BusinessPartner('{bid}')"
    
    params = {"sap-client": CLIENT}
    headers = {
        "X-CSRF-Token": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    r = s.patch(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        data=json.dumps(updates),
        timeout=60
    )
    
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"PATCH failed: {e}")
        try:
            print("Error:", json.dumps(r.json(), indent=2))
        except:
            print("Response:", r.text[:500])
        raise
    
    if r.status_code == 204:
        return {"message": "Updated successfully (no content returned)"}
    
    data = r.json()
    return (data.get("d") or data)
```

**Step-by-step breakdown:**

1. **Fetch CSRF Token**: `token, cookies = fetch_csrf(s)`
   - Gets a fresh CSRF token for this update operation
   - Captures session cookies for authentication

2. **Format Business Partner ID**: `bid = quote(bp10(bp_id), safe="")`
   - Ensures ID is 10 digits with leading zeros
   - URL-encodes the ID for safe use in URL

3. **Build URL**: `url = f"{API_BP}/A_BusinessPartner('{bid}')"`
   - Constructs the OData entity URL for the specific Business Partner
   - Uses the formatted and encoded ID

4. **Set Headers**:
   - `X-CSRF-Token`: Required for write operations
   - `Accept`: Specifies we want JSON response
   - `Content-Type`: Indicates we're sending JSON data
   - `Prefer: return=representation`: Requests updated entity in response

5. **Execute PATCH Request**:
   - Sends the update data as JSON
   - Includes CSRF token and cookies for authentication
   - Sets 60-second timeout

6. **Handle Response**:
   - `r.raise_for_status()` throws exception for HTTP errors
   - Handles both 200 OK (with data) and 204 No Content responses
   - Returns updated entity or confirmation message

**Usage Example:**
```python
# Update BusinessPartnerFullName
result = update_bp(
    session,
    "1000090",
    {"BusinessPartnerFullName": "Twister Updated"}
)
```

**What happens:**
- Business Partner with ID "1000090" gets updated
- Only the `BusinessPartnerFullName` field is modified
- Other fields remain unchanged (partial update)
- Returns confirmation or updated entity data

### 5. Update Sales Order Header Function

```python
def update_sales_order(s: requests.Session, so_number: str, updates: dict) -> dict:
    """
    Update Sales Order header fields.
    
    Args:
        s: Authenticated session
        so_number: Sales Order number
        updates: Dictionary of fields to update
    
    Returns:
        Updated entity or confirmation message
    """
    token, cookies = fetch_csrf(s)
    
    so_key = quote(so_number.zfill(10), safe="")
    url = f"{API_SO}/A_SalesOrder('{so_key}')"
    
    params = {"sap-client": CLIENT}
    headers = {
        "X-CSRF-Token": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    r = s.patch(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        data=json.dumps(updates),
        timeout=60
    )
    
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"PATCH failed: {e}")
        try:
            print("Error:", json.dumps(r.json(), indent=2))
        except:
            print("Response:", r.text[:500])
        raise
    
    if r.status_code == 204:
        return {"message": "Updated successfully"}
    
    data = r.json()
    return (data.get("d") or data)
```

**Step-by-step breakdown:**

1. **Fetch CSRF Token**: `token, cookies = fetch_csrf(s)`
   - Gets a fresh CSRF token for this update operation
   - Captures session cookies for authentication

2. **Format Sales Order Number**: `so_key = quote(so_number.zfill(10), safe="")`
   - Pads Sales Order number to 10 digits with leading zeros
   - URL-encodes the number for safe use in URL
   - Example: "1415" becomes "0000001415"

3. **Build URL**: `url = f"{API_SO}/A_SalesOrder('{so_key}')"`
   - Constructs the OData entity URL for the specific Sales Order
   - Uses the formatted and encoded Sales Order number

4. **Set Headers**:
   - `X-CSRF-Token`: Required for write operations
   - `Accept`: Specifies we want JSON response
   - `Content-Type`: Indicates we're sending JSON data
   - `Prefer: return=representation`: Requests updated entity in response

5. **Execute PATCH Request**:
   - Sends the update data as JSON
   - Includes CSRF token and cookies for authentication
   - Sets 60-second timeout

6. **Handle Response**:
   - `r.raise_for_status()` throws exception for HTTP errors
   - Handles both 200 OK (with data) and 204 No Content responses
   - Returns updated entity or confirmation message

**Usage Example:**
```python
# Update customer PO reference
result = update_sales_order(
    session,
    "1415",
    {
        "PurchaseOrderByCustomer": "CUST-PO-2024-001",
        "CustomerPurchaseOrderDate": sap_date(dt.date.today())
    }
)
```

**What happens:**
- Sales Order "1415" gets updated with customer purchase order information
- Only specified fields are modified (partial update)
- Other header fields remain unchanged
- Returns confirmation or updated Sales Order data

### 6. Update Sales Order Item Function

```python
def update_sales_order_item(
    s: requests.Session,
    so_number: str,
    item_number: str,
    updates: dict
) -> dict:
    """Update a specific Sales Order item."""
    token, cookies = fetch_csrf(s)
    
    so_key = quote(so_number.zfill(10), safe="")
    item_key = quote(item_number.zfill(6), safe="")
    url = f"{API_SO}/A_SalesOrderItem(SalesOrder='{so_key}',SalesOrderItem='{item_key}')"
    
    params = {"sap-client": CLIENT}
    headers = {
        "X-CSRF-Token": token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    r = s.patch(
        url,
        params=params,
        headers=headers,
        cookies=cookies,
        data=json.dumps(updates),
        timeout=60
    )
    
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"PATCH failed: {e}")
        raise
    
    if r.status_code == 204:
        return {"message": "Item updated successfully"}
    
    data = r.json()
    return (data.get("d") or data)
```

**Step-by-step breakdown:**

1. **Fetch CSRF Token**: `token, cookies = fetch_csrf(s)`
   - Gets a fresh CSRF token for this update operation
   - Captures session cookies for authentication

2. **Format Keys**:
   - `so_key = quote(so_number.zfill(10), safe="")`: Pads Sales Order number to 10 digits
   - `item_key = quote(item_number.zfill(6), safe="")`: Pads item number to 6 digits
   - Both are URL-encoded for safe use in URL

3. **Build URL**: `url = f"{API_SO}/A_SalesOrderItem(SalesOrder='{so_key}',SalesOrderItem='{item_key}')"`
   - Constructs the OData entity URL for the specific Sales Order item
   - Uses composite key with both Sales Order and Item numbers
   - Example: Sales Order "1415", Item "10" becomes "A_SalesOrderItem(SalesOrder='0000001415',SalesOrderItem='000010')"

4. **Set Headers**:
   - `X-CSRF-Token`: Required for write operations
   - `Accept`: Specifies we want JSON response
   - `Content-Type`: Indicates we're sending JSON data
   - `Prefer: return=representation`: Requests updated entity in response

5. **Execute PATCH Request**:
   - Sends the update data as JSON
   - Includes CSRF token and cookies for authentication
   - Sets 60-second timeout

6. **Handle Response**:
   - `r.raise_for_status()` throws exception for HTTP errors
   - Handles both 200 OK (with data) and 204 No Content responses
   - Returns updated entity or confirmation message

**Usage Example:**
```python
# Update item quantity
result = update_sales_order_item(
    session,
    "1415",
    "10",
    {"RequestedQuantity": "5"}
)
```

**What happens:**
- Sales Order "1415", Item "10" gets updated with new quantity
- Only specified fields are modified (partial update)
- Other item fields remain unchanged
- Returns confirmation or updated item data

### 7. Main Function and Complete Workflow

```python
def main():
    """Demonstrate updating data with PATCH."""
    print("="*60)
    print("Chapter 06: Updating Data (PATCH Operations)")
    print("="*60)
    
    session = sess()
    
    # Example 1: Update Business Partner
    print("\n=== Update Business Partner (Demo) ===")
    print("Note: BP updates may fail if field is read-only or BP is locked")
    print("This is for demonstration purposes only.")
    
    # Uncomment to actually update (be careful!)
    # try:
    #     result = update_bp(
    #         session,
    #         "1000090",
    #         {"BusinessPartnerFullName": "Twister Updated"}
    #     )
    #     print(f"Update result: {result}")
    #     
    #     # Verify
    #     bp = get_bp(session, "1000090")
    #     print(f"New name: {bp.get('BusinessPartnerFullName')}")
    # except Exception as e:
    #     print(f"Update failed: {e}")
    
    # Example 2: Update Sales Order
    print("\n=== Update Sales Order (Demo) ===")
    print("Note: SO updates may fail based on status and authorizations")
    print("This demonstrates the PATCH operation structure.")
    
    # Uncomment to actually update (be careful!)
    # try:
    #     result = update_sales_order(
    #         session,
    #         "1415",
    #         {"PurchaseOrderByCustomer": "PO-2024-TEST"}
    #     )
    #     print(f"Update result: {result}")
    # except Exception as e:
    #     print(f"Update failed: {e}")
    
    print("\nIMPORTANT NOTES:")
    print("- PATCH operations modify existing data")
    print("- Always fetch a fresh CSRF token")
    print("- Test updates in development environment first")
    print("- Not all fields are updatable")
    print("- Document status may prevent updates")
    
    print("\n" + "="*60)
    print("Chapter 06 completed!")
    print("Next: Chapter 07 - Batch Operations")
    print("="*60)
```

**How the complete workflow works:**

1. **Session Creation**: `session = sess()`
   - Creates authenticated session with S/4HANA
   - Sets up basic authentication and default headers

2. **Safety Measures**:
   - All update operations are commented out by default
   - Demonstrates the structure without actually modifying data
   - Includes error handling for failed updates

3. **Update Process Flow**:
   - Each update function follows the same pattern:
     - Fetch CSRF token
     - Format entity keys
     - Build OData URL
     - Set required headers
     - Execute PATCH request
     - Handle response

4. **Error Handling**:
   - `try/except` blocks catch and display errors
   - Detailed error messages help with troubleshooting
   - Graceful handling of different response codes

5. **Verification**:
   - After updates, data can be re-read to verify changes
   - Confirmation messages indicate successful operations

## Expected Output

```
=== Update Business Partner ===
Updating BP 1000090...
Result: {"message": "Updated successfully (no content returned)"}

=== Update Sales Order Header ===
Updating SO 1415...
Updated: PurchaseOrderByCustomer = CUST-PO-2024-001

=== Update Sales Order Item ===
Updating item 10 in SO 1415...
Item quantity updated to 5
```

## Updatable Fields

### Business Partner
- `BusinessPartnerFullName`
- `SearchTerm1`, `SearchTerm2`
- `BusinessPartnerIsBlocked`
- Custom fields (Z-fields)

### Sales Order Header
- `PurchaseOrderByCustomer`
- `CustomerPurchaseOrderDate`
- `RequestedDeliveryDate`
- `ShippingCondition`
- `YourReference` (custom field)

### Sales Order Items
- `RequestedQuantity`
- `RequestedDeliveryDate`
- `ItemDescription` (if allowed by customizing)

**Note:** Many fields are read-only or controlled by S/4HANA workflow. Always check field metadata.

## Common Update Issues

### Issue 1: Field is Read-Only (400)
**Solution:** Check API documentation for updatable fields

### Issue 2: Update Not Allowed Due to Status
**Solution:** Some fields can't be changed after order processing starts

### Issue 3: Validation Error
**Solution:** Ensure updated values meet business rules (e.g., delivery date after order date)

## Key Takeaways

### Technical Concepts

1. **PATCH vs PUT**: PATCH modifies only specified fields, while PUT would replace the entire entity
2. **CSRF Token Management**: Always fetch a fresh CSRF token for each update operation
3. **Response Handling**: Handle both 200 OK (with data) and 204 No Content responses
4. **Partial Updates**: Only the fields you specify get updated, others remain unchanged
5. **Field Restrictions**: Not all fields are updatable due to business rules and system constraints

### Best Practices

1. **Safety First**: Test updates in development environment before production
2. **Error Handling**: Always include proper error handling and logging
3. **Verification**: Re-read entities after updates to confirm changes
4. **Documentation**: Check field metadata to understand which fields are updatable
5. **Status Awareness**: Document status may prevent certain updates

### Common Patterns

1. **Session Management**: Reuse authenticated sessions for multiple operations
2. **Key Formatting**: Always format entity keys (pad with zeros, URL-encode)
3. **Header Configuration**: Include all required headers for write operations
4. **Timeout Handling**: Set appropriate timeouts for network operations
5. **Response Processing**: Handle different response codes appropriately

## Understanding the Code Structure

### Function Hierarchy
```
main()
├── sess() - Session creation and authentication
├── fetch_csrf() - CSRF token management
├── bp10() - Business Partner ID formatting
├── update_bp() - Business Partner updates
├── update_sales_order() - Sales Order header updates
└── update_sales_order_item() - Sales Order item updates
```

### Data Flow
1. **Authentication**: Create session with credentials
2. **Token Fetch**: Get CSRF token for write operations
3. **Key Formatting**: Format entity keys for URL construction
4. **Request Building**: Construct OData URLs and headers
5. **Execution**: Send PATCH request with update data
6. **Response Handling**: Process success/error responses
7. **Verification**: Optionally re-read data to confirm changes

## Next Steps

Proceed to **Chapter 07: Batch Operations** to learn how to perform multiple operations in a single HTTP request, which is more efficient for bulk updates.

## Files in This Chapter

- `README.md` - This comprehensive guide with step-by-step explanations
- `update_data.py` - Complete example script with all update functions
- `config.py` - Configuration file with connection settings

