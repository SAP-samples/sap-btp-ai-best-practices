"""
Utility functions for handling vendor identification and duplicate naming.
"""

def ensure_unique_vendor_ids(vendor_ids):
    """
    Ensure vendor IDs are unique by appending incremental integers to duplicates.
    
    Args:
        vendor_ids (list or pandas.Series): List of vendor IDs that may contain duplicates
        
    Returns:
        list: List of unique vendor IDs with incremental suffixes for duplicates
        
    Example:
        input: ['Supplier A-US', 'Supplier A-US', 'Supplier B-CA', 'Supplier A-US']
        output: ['Supplier A-US-1', 'Supplier A-US-2', 'Supplier B-CA', 'Supplier A-US-3']
    """
    # Convert to list if pandas Series
    if hasattr(vendor_ids, 'tolist'):
        vendor_ids = vendor_ids.tolist()
    
    # Count occurrences of each ID
    id_counts = {}
    for vendor_id in vendor_ids:
        id_counts[vendor_id] = id_counts.get(vendor_id, 0) + 1
    
    # Find duplicates
    duplicates = {vendor_id: count for vendor_id, count in id_counts.items() if count > 1}
    
    if not duplicates:
        # No duplicates found, return original list
        return list(vendor_ids)
    
    # Create unique IDs with incremental suffixes
    id_counters = {}
    unique_ids = []
    
    for vendor_id in vendor_ids:
        if vendor_id in duplicates:
            # This ID has duplicates, so we need to add a suffix
            if vendor_id not in id_counters:
                id_counters[vendor_id] = 1
            else:
                id_counters[vendor_id] += 1
            
            unique_ids.append(f"{vendor_id}-{id_counters[vendor_id]}")
        else:
            # This ID is unique, keep it as is
            unique_ids.append(vendor_id)
    
    return unique_ids


def create_vendor_plant_id(name1, land1):
    """
    Create a vendor-plant identifier from NAME1 and LAND1 fields.
    
    Args:
        name1 (str): Vendor name
        land1 (str): Country/plant code
        
    Returns:
        str: Formatted vendor-plant ID
    """
    if not land1 or str(land1).lower() in ['nan', 'none', '']:
        land1 = 'XX'
    
    return f"{name1}-{land1}"