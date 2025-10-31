import pandas as pd
import os

def get_country_tariff_stats_from_file(file_path):
    """
    Get tariff stats directly from a CSV file to validate values.
    Used for debugging tariff-related issues.
    """
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"Loaded file: {file_path} with {len(df)} rows")
        
        # Check if required columns exist
        required_cols = ['LIFNR', 'LAND1', 'TariffImpact_raw_percent']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Group by country and calculate stats
        stats = df.groupby('LAND1').agg(
            AvgTariff=('TariffImpact_raw_percent', 'mean'),
            MinTariff=('TariffImpact_raw_percent', 'min'),
            MaxTariff=('TariffImpact_raw_percent', 'max'),
            Count=('LIFNR', 'nunique')
        ).reset_index()
        
        # Print stats for debugging
        print("Tariff statistics by country:")
        for _, row in stats.iterrows():
            print(f"  {row['LAND1']}: Avg={row['AvgTariff']:.2f}%, Min={row['MinTariff']:.2f}%, Max={row['MaxTariff']:.2f}%, Count={row['Count']}")
        
        return stats
    
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        return None

# This can be run directly for debugging
if __name__ == "__main__":
    file_path = os.path.join('tables', 'vendor_maktx_ranking_tariff_values.csv')
    get_country_tariff_stats_from_file(file_path)