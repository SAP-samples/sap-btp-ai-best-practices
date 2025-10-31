import json
import random
import os

# Country distance factors from US perspective (0-1 scale, where 1 is furthest)
# Based on actual countries in the data
COUNTRY_DISTANCE_FACTORS = {
    'CN': 1.0,    # China - Pacific shipping
    'DE': 0.8,    # Germany - Atlantic shipping
    'HU': 0.85,   # Hungary - Central Europe
    'IN': 0.9,    # India - Via Suez/Pacific
    'ID': 1.0,    # Indonesia - Southeast Asia
    'MY': 0.95,   # Malaysia - Southeast Asia
    'MX': 0.3,    # Mexico - NAFTA neighbor
    'TH': 0.95,   # Thailand - Southeast Asia
    'US': 0.1,    # United States - Domestic shipping
}

# Material-specific logistics complexity factors
# Based on weight, size, handling requirements
MATERIAL_LOGISTICS_FACTORS = {
    'ECR-SENSOR': 0.3,      # Small, lightweight, air-shippable
    'ECR-ZFRAME': 0.8,      # Large frame, container shipping
    'EMN-BRAKES': 0.6,      # Medium weight, special handling
    'EMN-HANDLE': 0.4,      # Medium size, standard shipping
    'EMN-MOTOR': 0.9,       # Heavy, requires special transport
    'EMN-THROTTLE': 0.35,   # Small electronic component
}

# Base logistics cost as percentage of unit price
BASE_LOGISTICS_COST_RATE = 0.15  # 15% of unit price as baseline

def generate_logistics_cost_factors():
    """
    Generate logistics cost factors for each country-material combination.
    
    The logistics cost will be calculated as:
    logistics_cost = unit_price * BASE_RATE * country_factor * material_factor * random_factor
    
    Where random_factor is between 0.8 and 1.2 (±20% variation)
    """
    print("Generating logistics cost factors...")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(script_dir, '..', 'tables')
    
    # Create logistics factors for each combination
    logistics_data = {
        'country_factors': COUNTRY_DISTANCE_FACTORS,
        'material_factors': MATERIAL_LOGISTICS_FACTORS,
        'base_rate': BASE_LOGISTICS_COST_RATE,
        'combinations': {}
    }
    
    # Generate combination factors
    for country_code, distance_factor in COUNTRY_DISTANCE_FACTORS.items():
        for material, material_factor in MATERIAL_LOGISTICS_FACTORS.items():
            # Add some randomness (±20%)
            random_factor = random.uniform(0.8, 1.2)
            
            # Calculate combined logistics factor
            combined_factor = distance_factor * material_factor * random_factor
            
            # Store in format: "COUNTRY_MATERIAL": factor
            key = f"{country_code}_{material}"
            logistics_data['combinations'][key] = round(combined_factor, 4)
    
    # Add default values for unknown combinations
    logistics_data['defaults'] = {
        'country_factor': 0.7,
        'material_factor': 0.5,
        'combined_factor': 0.35
    }
    
    # Save to JSON file
    output_path = os.path.join(tables_dir, 'logistics_factors.json')
    with open(output_path, 'w') as f:
        json.dump(logistics_data, f, indent=2)
    
    print(f"Generated logistics factors and saved to: {output_path}")
    print(f"Total combinations: {len(logistics_data['combinations'])}")
    
    # Print summary statistics
    print("\nLogistics Cost Summary:")
    print(f"Base rate: {BASE_LOGISTICS_COST_RATE * 100:.1f}% of unit price")
    print("\nCountry distance factors:")
    for country, factor in sorted(COUNTRY_DISTANCE_FACTORS.items()):
        print(f"  {country}: {factor:.2f}")
    
    print("\nMaterial complexity factors:")
    for material, factor in sorted(MATERIAL_LOGISTICS_FACTORS.items()):
        print(f"  {material}: {factor:.2f}")
    
    # Show example calculations
    print("\nExample logistics cost multipliers (base_rate * factors):")
    examples = [
        ("US", "ECR-SENSOR"),
        ("CN", "EMN-MOTOR"),
        ("MX", "EMN-HANDLE"),
        ("DE", "ECR-ZFRAME")
    ]
    
    for country, material in examples:
        key = f"{country}_{material}"
        if key in logistics_data['combinations']:
            total_factor = BASE_LOGISTICS_COST_RATE * logistics_data['combinations'][key]
            print(f"  {country} + {material}: {total_factor:.4f} ({total_factor*100:.2f}% of unit price)")
    
    return logistics_data

if __name__ == "__main__":
    logistics_data = generate_logistics_cost_factors()
    
    # Verify the data can be loaded
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.join(script_dir, '..', 'tables')
    output_path = os.path.join(tables_dir, 'logistics_factors.json')
    
    with open(output_path, 'r') as f:
        loaded_data = json.load(f)
    
    print(f"\nVerification: Successfully loaded {len(loaded_data['combinations'])} combinations from JSON")