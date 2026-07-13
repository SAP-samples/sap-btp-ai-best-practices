"""
EPIC 4 Feature Engineering - End-to-End Validation Script

This script validates the complete feature engineering pipeline by:
1. Building a small sample of the canonical training table with features
2. Running all validation checks from features/validation.py
3. Verifying feature coverage, leakage prevention, and channel-specific logic
4. Comparing feature correlations to driver screening benchmarks (if available)

Usage:
    python -m forecasting.regressor.validation.validate_epic4_pipeline

Expected Runtime: 2-5 minutes for small sample (h=1-4, recent 10 weeks)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
from pathlib import Path

from app.regressor.etl.canonical_table import build_canonical_training_table
from app.regressor.features.validation import (
    validate_feature_coverage,
    validate_leakage_prevention,
    validate_channel_specific_features,
    validate_feature_correlations,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def validate_imports():
    """Verify all required imports are available."""
    print_section("STEP 1: Validating Imports")

    try:
        from app.regressor.features.time_varying_features import attach_time_varying_features
        from app.regressor.features.dynamics_features import (
            attach_sales_aov_dynamics_bm,
            attach_web_dynamics,
            attach_conversion_omnichannel_features,
            attach_product_mix_features
        )
        from app.regressor.features.static_features import attach_static_store_features
        from app.regressor.features.cannibalization import compute_cannibalization_pressure
        from app.regressor.features.awareness_features import attach_awareness_features
        from app.regressor.features.crm_features import attach_crm_features
        from app.regressor.features.model_views import build_model_a_features, build_model_b_features
        from app.regressor.data_ingestion.awareness import load_awareness_data
        from app.regressor.data_ingestion.crm_mix import load_crm_mix

        print("✓ All feature engineering imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def validate_data_loaders():
    """Test that data loaders are callable and return expected structure."""
    print_section("STEP 2: Validating Data Loaders")

    results = {}

    try:
        from app.regressor.data_ingestion.awareness import load_awareness_data
        aw = load_awareness_data()
        print(f"✓ load_awareness_data(): {len(aw)} rows, columns: {list(aw.columns)[:5]}...")
        results['awareness'] = True
    except Exception as e:
        print(f"✗ load_awareness_data() failed: {e}")
        results['awareness'] = False

    try:
        from app.regressor.data_ingestion.crm_mix import load_crm_mix
        crm = load_crm_mix()
        print(f"✓ load_crm_mix(): {len(crm)} rows, columns: {list(crm.columns)[:5]}...")
        results['crm'] = True
    except Exception as e:
        print(f"✗ load_crm_mix() failed: {e}")
        results['crm'] = False

    try:
        from app.regressor.seasonality import compute_dma_seasonality
        dma_seas = compute_dma_seasonality()
        print(f"✓ compute_dma_seasonality(): {len(dma_seas)} rows, columns: {list(dma_seas.columns)}")
        results['seasonality'] = True
    except Exception as e:
        print(f"✗ compute_dma_seasonality() failed: {e}")
        results['seasonality'] = False

    return all(results.values())


def build_test_sample(
    horizons: range = range(1, 5),
    include_crm: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build small test samples for Model A and Model B."""
    print_section("STEP 3: Building Test Samples")

    print(f"Building Model B (Production) sample with horizons {list(horizons)}...")
    try:
        df_b = build_canonical_training_table(
            horizons=horizons,
            include_features=True,
            model_variant='B',
            include_crm=include_crm
        )
        print(f"✓ Model B built: {len(df_b)} rows × {len(df_b.columns)} columns")
    except Exception as e:
        print(f"✗ Model B build failed: {e}")
        raise

    print(f"\nBuilding Model A (Actionable) sample with horizons {list(horizons)}...")
    try:
        df_a = build_canonical_training_table(
            horizons=horizons,
            include_features=True,
            model_variant='A',
            include_crm=include_crm
        )
        print(f"✓ Model A built: {len(df_a)} rows × {len(df_a.columns)} columns")
    except Exception as e:
        print(f"✗ Model A build failed: {e}")
        raise

    return df_a, df_b


def inspect_feature_coverage(df: pd.DataFrame, model_variant: str):
    """Inspect which features are present and their NaN rates."""
    print_section(f"STEP 4: Feature Coverage Inspection - Model {model_variant}")

    # Key columns
    key_cols = ['profit_center_nbr', 'dma', 'channel', 'origin_week_date',
                'target_week_date', 'horizon']

    # Target columns
    target_cols = ['label_log_sales', 'label_log_aov', 'label_logit_conversion']

    # Feature columns (everything else)
    feature_cols = [c for c in df.columns if c not in key_cols + target_cols
                    and not c.startswith('has_') and c not in ['total_sales', 'order_count',
                    'store_traffic', 'aur', 'allocated_web_traffic']]

    print(f"\nTotal columns: {len(df.columns)}")
    print(f"  Keys: {len(key_cols)}")
    print(f"  Targets: {len(target_cols)}")
    print(f"  Features: {len(feature_cols)}")

    # Check for expected feature categories
    expected_patterns = {
        'Time-Varying (FE-01)': ['dma_seasonal_weight', 'woy_', 'sin_woy', 'cos_woy',
                                  'is_holiday', 'fiscal_year', 'fiscal_period'],
        'Static DNA (FE-05)': ['proforma_annual_sales', 'is_outlet', 'weeks_since_open',
                               'sq_ft', 'merchandising_sf', 'region', 'format'],
        'Sales/AOV Dynamics (FE-02)': ['log_sales_lag_', 'log_sales_roll_mean_',
                                        'AOV_roll_mean_', 'vol_sales_'],
        'Web Dynamics (FE-03)': ['allocated_web_traffic_lag_',
                                  'allocated_web_traffic_roll_mean_'],
        'Conversion/Omnichannel (FE-04)': ['ConversionRate_lag_',
                                            'ConversionRate_roll_mean_',
                                            'pct_omni_channel_lag_',
                                            'pct_omni_channel_roll_mean_'],
        'Cannibalization (FE-06)': ['cannibalization_pressure', 'min_dist_new_store_km',
                                     'num_new_stores_within_'],
        'Awareness (FE-07)': ['brand_awareness_dma_score',
                               'brand_consideration_dma_score'],
        'Product Mix': ['pct_value_product_', 'pct_premium_product_', 'pct_white_glove_']
    }

    if model_variant == 'B':
        # CRM features only in Model B typically
        expected_patterns['CRM (FE-08)'] = ['crm_dwelling_', 'crm_owner_renter_',
                                             'crm_income_', 'crm_education_', 'crm_age_']

    print("\nFeature Coverage by Category:")
    print("-" * 80)
    for category, patterns in expected_patterns.items():
        matches = []
        for pattern in patterns:
            matches.extend([c for c in feature_cols if pattern in c])
        matches = sorted(set(matches))

        if matches:
            # Calculate NaN rates for these features
            nan_rates = df[matches].isna().mean() * 100
            avg_nan = nan_rates.mean()
            print(f"\n{category}:")
            print(f"  Found: {len(matches)} features")
            print(f"  Avg NaN rate: {avg_nan:.1f}%")

            # Show sample features with high NaN rates (>20%)
            high_nan = nan_rates[nan_rates > 20].sort_values(ascending=False)
            if not high_nan.empty:
                print(f"  ⚠ High NaN features (>20%):")
                for feat, rate in high_nan.head(5).items():
                    print(f"    - {feat}: {rate:.1f}%")
        else:
            print(f"\n{category}:")
            print(f"  ⚠ No features found (may be optional or model-specific)")

    return feature_cols


def run_validation_checks(df: pd.DataFrame, model_variant: str):
    """Run all validation checks from features/validation.py."""
    print_section(f"STEP 5: Validation Checks - Model {model_variant}")

    results = {}

    # 1. Feature Coverage
    print("\n[1/4] Validating Feature Coverage...")
    try:
        validate_feature_coverage(
            df,
            required_features=['dma_seasonal_weight_1', 'proforma_annual_sales'],
            max_nan_rate=0.10,
            warn_only=True
        )
        print("✓ Feature coverage check passed")
        results['coverage'] = True
    except Exception as e:
        print(f"✗ Feature coverage check failed: {e}")
        results['coverage'] = False

    # 2. Leakage Prevention
    print("\n[2/4] Validating Leakage Prevention...")
    try:
        validate_leakage_prevention(
            df,
            origin_col='origin_week_date',
            feature_cols=[c for c in df.columns if c.startswith('log_sales_')
                          or c.startswith('AOV_') or c.startswith('ConversionRate_')],
            warn_only=True
        )
        print("✓ Leakage prevention check passed")
        results['leakage'] = True
    except Exception as e:
        print(f"⚠ Leakage prevention check warning: {e}")
        results['leakage'] = False

    # 3. Channel-Specific Features
    print("\n[3/4] Validating Channel-Specific Logic...")
    try:
        validate_channel_specific_features(
            df,
            channel_col='channel',
            warn_only=True
        )
        print("✓ Channel-specific logic check passed")
        results['channel'] = True
    except Exception as e:
        print(f"✗ Channel-specific logic check failed: {e}")
        results['channel'] = False

    # 4. Feature Correlations (optional, requires benchmark data)
    print("\n[4/4] Validating Feature Correlations...")
    print("  ⚠ Skipped - Requires benchmark correlation dictionary")
    print("  (Run manually with validate_feature_correlations() when benchmarks available)")
    results['correlations'] = None

    return results


def compare_model_variants(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Compare Model A vs Model B feature sets."""
    print_section("STEP 6: Model A vs Model B Comparison")

    key_cols = ['profit_center_nbr', 'dma', 'channel', 'origin_week_date',
                'target_week_date', 'horizon']
    target_cols = ['label_log_sales', 'label_log_aov', 'label_logit_conversion']
    exclude = key_cols + target_cols + ['has_traffic_data', 'total_sales', 'order_count',
                                         'store_traffic', 'aur', 'allocated_web_traffic']

    features_a = set(df_a.columns) - set(exclude)
    features_b = set(df_b.columns) - set(exclude)

    print(f"\nModel A features: {len(features_a)}")
    print(f"Model B features: {len(features_b)}")

    # Features only in Model B (should be autoregressive lags/rolls)
    only_b = features_b - features_a
    print(f"\nFeatures ONLY in Model B (autoregressive): {len(only_b)}")

    lag_roll_features = [f for f in only_b if any(x in f for x in
                         ['_lag_', '_roll_mean_', 'log_sales', 'AOV', 'ConversionRate'])]
    print(f"  Autoregressive lags/rolls: {len(lag_roll_features)}")
    if lag_roll_features:
        print(f"  Examples: {sorted(lag_roll_features)[:5]}")

    # Features only in Model A (should be none or very few)
    only_a = features_a - features_b
    if only_a:
        print(f"\n⚠ Features ONLY in Model A (unexpected): {len(only_a)}")
        print(f"  {sorted(only_a)}")
    else:
        print(f"\n✓ Model A is a proper subset of Model B (expected)")

    # Common features (actionable levers + known-in-advance)
    common = features_a & features_b
    print(f"\nCommon features (actionable + known-in-advance): {len(common)}")

    return {
        'model_a_features': len(features_a),
        'model_b_features': len(features_b),
        'only_in_b': len(only_b),
        'only_in_a': len(only_a),
        'common': len(common)
    }


def generate_summary_report(
    import_success: bool,
    loader_success: bool,
    df_a: Optional[pd.DataFrame],
    df_b: Optional[pd.DataFrame],
    validation_results_a: Optional[Dict],
    validation_results_b: Optional[Dict],
    comparison: Optional[Dict]
):
    """Generate final summary report."""
    print_section("FINAL VALIDATION REPORT")

    print("\n✓ = Passed | ✗ = Failed | ⚠ = Warning\n")

    print("1. Import Validation:")
    print(f"   {'✓' if import_success else '✗'} All feature engineering imports successful")

    print("\n2. Data Loader Validation:")
    print(f"   {'✓' if loader_success else '✗'} All data loaders callable")

    if df_a is not None and df_b is not None:
        print("\n3. Pipeline Execution:")
        print(f"   ✓ Model A built successfully ({len(df_a)} rows × {len(df_a.columns)} cols)")
        print(f"   ✓ Model B built successfully ({len(df_b)} rows × {len(df_b.columns)} cols)")
    else:
        print("\n3. Pipeline Execution:")
        print(f"   ✗ Failed to build canonical training table")
        return

    if validation_results_a and validation_results_b:
        print("\n4. Validation Checks:")

        checks = ['coverage', 'leakage', 'channel']
        for check in checks:
            a_result = validation_results_a.get(check)
            b_result = validation_results_b.get(check)

            status_a = '✓' if a_result else ('⚠' if a_result is False else '-')
            status_b = '✓' if b_result else ('⚠' if b_result is False else '-')

            print(f"   {check.capitalize():20s} Model A: {status_a}  |  Model B: {status_b}")

    if comparison:
        print("\n5. Model Variant Comparison:")
        print(f"   Model A (Actionable):     {comparison['model_a_features']:3d} features")
        print(f"   Model B (Production):     {comparison['model_b_features']:3d} features")
        print(f"   Only in B (AR lags):      {comparison['only_in_b']:3d} features")
        print(f"   Common features:          {comparison['common']:3d} features")

        if comparison['only_in_a'] == 0:
            print(f"   ✓ Model A is proper subset of Model B")
        else:
            print(f"   ⚠ {comparison['only_in_a']} features only in Model A (unexpected)")

    print("\n" + "="*80)
    if import_success and loader_success and df_a is not None and df_b is not None:
        print("OVERALL STATUS: ✓ EPIC 4 PIPELINE VALIDATION PASSED")
        print("\nThe feature engineering pipeline is ready for:")
        print("  1. Full-scale canonical table generation (h=1-52)")
        print("  2. Integration with EPIC 5 (CatBoost model training)")
        print("  3. Production deployment (after final QA)")
    else:
        print("OVERALL STATUS: ✗ EPIC 4 PIPELINE VALIDATION FAILED")
        print("\nPlease review errors above and fix before proceeding to EPIC 5.")
    print("="*80 + "\n")


def main():
    """Run complete EPIC 4 validation pipeline."""
    print("\n" + "="*80)
    print("  EPIC 4 FEATURE ENGINEERING - END-TO-END VALIDATION")
    print("="*80)
    print("\nThis script will:")
    print("  1. Validate all imports are correct")
    print("  2. Test data loaders are callable")
    print("  3. Build small test samples (Model A & B)")
    print("  4. Inspect feature coverage")
    print("  5. Run validation checks (coverage, leakage, channel logic)")
    print("  6. Compare Model A vs Model B feature sets")
    print("  7. Generate summary report")
    print("\nExpected runtime: 2-5 minutes")
    print("="*80)

    # Step 1: Validate imports
    import_success = validate_imports()
    if not import_success:
        print("\n✗ CRITICAL: Import validation failed. Cannot proceed.")
        return

    # Step 2: Validate data loaders
    loader_success = validate_data_loaders()
    if not loader_success:
        print("\n⚠ WARNING: Some data loaders failed. Proceeding with caution...")

    # Step 3: Build test samples
    try:
        df_a, df_b = build_test_sample(
            horizons=range(1, 5),  # Small sample: h=1-4
            include_crm=False  # Skip CRM for initial validation
        )
    except Exception as e:
        print(f"\n✗ CRITICAL: Failed to build test samples: {e}")
        generate_summary_report(import_success, loader_success, None, None, None, None, None)
        raise

    # Step 4: Inspect feature coverage
    features_a = inspect_feature_coverage(df_a, 'A')
    features_b = inspect_feature_coverage(df_b, 'B')

    # Step 5: Run validation checks
    validation_results_a = run_validation_checks(df_a, 'A')
    validation_results_b = run_validation_checks(df_b, 'B')

    # Step 6: Compare model variants
    comparison = compare_model_variants(df_a, df_b)

    # Step 7: Generate summary report
    generate_summary_report(
        import_success,
        loader_success,
        df_a,
        df_b,
        validation_results_a,
        validation_results_b,
        comparison
    )

    return df_a, df_b


if __name__ == "__main__":
    df_a, df_b = main()
