# Validation Module

## Overview

This module contains comprehensive runtime validation scripts for Company X Sales Forecasting Model. The validation framework ensures data quality, feature engineering correctness, and prevents common ML pitfalls such as data leakage and target contamination throughout the forecasting pipeline.

The validation scripts follow patterns established in `prototype/validate_runtime.py` and align with checkpoints defined in the parent `PROJECT_PLAN.md`. Each script performs rigorous checks across different stages of the model development lifecycle.

## Context: Model A vs Model B

Company X Sales Forecasting System maintains two model variants:

- **Model A (Actionable/Explainability)**: Contains only business-actionable levers and known-in-advance features. Used for SHAP analysis and stakeholder communication. Ensures that explanations focus on controllable factors rather than autoregressive patterns.

- **Model B (Production/Prediction)**: Full feature set including autoregressive lags and rolling statistics for maximum predictive accuracy. Used for production forecasts.

This dual-model architecture prevents autoregressive features from overshadowing operational levers in explainability outputs while maintaining production forecast quality.

## File Inventory

| File | Lines | Purpose | Epic Coverage |
|------|-------|---------|---------------|
| `validate_epic1_epic2.py` | 1,054 | Validates data ingestion pipelines (Epic 1) and seasonality artifacts (Epic 2). Checks all data loaders, traffic missingness flags, awareness feeds, CRM demographics, DMA seasonal weights, and sister-DMA fallback logic. | Epic 1, Epic 2 |
| `validate_epic4_pipeline.py` | 454 | End-to-end validation of feature engineering pipeline (Epic 4). Builds small canonical training table samples for both Model A and Model B, validates feature coverage, leakage prevention, channel-specific logic, and compares model variants. | Epic 4 |
| `validate_sister_store.py` | 73 | Validates sister-store mapping for stores with insufficient history (<3 years). Ensures distance calculations are finite, preference logic (same DMA > same region > nearest) is followed, and all low-history stores receive mappings. | Epic 2 (SEAS-02) |
| `__init__.py` | 7 | Package initialization. Provides module-level documentation. | N/A |

## Validation Coverage by Epic

### Epic 1: Data Ingestion & Cleaning

**Script:** `validate_epic1_epic2.py`

**Checkpoints:**
- **DATA-01**: Port and verify data loaders
  - Validates `load_written_sales()` converts "NULL" strings to NaN
  - Checks percentage columns are non-negative
  - Verifies 163 unique stores present
  - Validates `load_store_master()` parses proforma_annual_sales and is_outlet correctly
  - Checks latitude/longitude coordinates are within reasonable US ranges

- **DATA-02**: Traffic missingness flag
  - Validates `has_traffic_data` flag computation
  - Ensures stores with >20% missing traffic are flagged as 0
  - Verifies stores with <8 non-null weeks are flagged as 0
  - Confirms flag successfully merged to sales data

- **DATA-03**: Awareness/Consideration feeds
  - Validates 38,000+ rows loaded from refreshed export
  - Checks Market to DMA mapping applied correctly (96.7% coverage)
  - Ensures awareness values are numeric
  - Verifies date range covers 2022-2025
  - Validates forward-fill logic for monthly to weekly conversion

- **DATA-04**: CRM demographics
  - Validates function returns tuple of 2 DataFrames (static, crm)
  - Checks static: 163 stores with 1 row each
  - Verifies CRM: approximately 96 rows per store
  - Ensures 23 CRM percentage features present
  - Validates lag/roll variants available (lag_1, lag_4, roll_4, roll_8)

### Epic 2: Layer 0 Artifacts (Seasonality & Baselines)

**Script:** `validate_epic1_epic2.py`

**Checkpoints:**
- **SEAS-01**: DMA seasonal weight computation
  - Validates 52 weights per DMA
  - Checks weights sum to 1.0 (normalized)
  - Ensures week 53 folded into week 1
  - Verifies 3-point cyclic smoothing applied
  - Validates all weights are positive
  - Checks smoothness (max week-to-week change <0.01)

- **SEAS-02**: Sister-DMA fallback
  - Identifies DMAs with <3 years history
  - Validates nearest same-region DMA chosen as sister
  - Verifies fallback to nearest overall if no region match
  - Checks distance calculations are reasonable (<1000 miles average)
  - Ensures sister mapping applied in seasonality computation
  - Validates >50% same-region preference achieved

**Script:** `validate_sister_store.py`

**Checks:**
- Store-level fallback for short histories (<3 years by default)
- Distance calculations finite and reasonable
- Preference hierarchy: same DMA > same region > nearest overall
- Coverage: all low-history stores receive sister mapping

### Epic 4: Feature Engineering

**Script:** `validate_epic4_pipeline.py`

**Validation Steps:**
1. **Import Validation**: Verifies all feature engineering modules can be imported
2. **Data Loader Validation**: Tests that loaders are callable and return expected structures
3. **Pipeline Execution**: Builds small test samples (horizons 1-4) for Model A and Model B
4. **Feature Coverage Inspection**: Analyzes which features are present and their NaN rates by category:
   - Time-Varying (FE-01): seasonal weights, week-of-year, holidays
   - Static DNA (FE-05): proforma_annual_sales, is_outlet, store age
   - Sales/AOV Dynamics (FE-02): lags, rolling means, volatility
   - Web Dynamics (FE-03): allocated_web_traffic features
   - Conversion/Omnichannel (FE-04): conversion rate features
   - Cannibalization (FE-06): pressure metrics, distance to new stores
   - Awareness (FE-07): brand awareness and consideration scores
   - Product Mix: value/premium/white-glove percentages
   - CRM (FE-08): demographic mix features (Model B only typically)
5. **Validation Checks**: Runs checks from `features/validation.py`
   - Feature coverage check
   - Leakage prevention check
   - Channel-specific logic check
6. **Model Comparison**: Compares Model A vs Model B feature sets
   - Validates Model A is proper subset of Model B
   - Identifies autoregressive features only in Model B
7. **Summary Report**: Generates comprehensive validation report

## Key Validation Metrics & Methods

### Data Quality Metrics
- **Completeness**: Row counts, unique store/DMA counts, date range coverage
- **Validity**: Data type checks, range validation (e.g., percentages 0-100%, coordinates within US)
- **Consistency**: Cross-table joins, flag computation logic verification

### Feature Engineering Metrics
- **Coverage**: Percentage of expected features present, NaN rates by category
- **Leakage Prevention**: Temporal checks ensuring features at time t0 don't contain information from t0+h
- **Channel Specificity**: Web-only features (allocated_web_traffic) null for B&M channel
- **Normalization**: Seasonal weights sum to 1.0 per DMA, CRM percentages valid

### Cross-Validation Strategies
- **Recomputation Validation**: Independently recomputes key metrics (e.g., traffic flags) and compares to original
- **Sister Mapping Validation**: Verifies fallback logic by checking same-region preference rates and distance distributions
- **Feature Correlation Benchmarks**: Compares engineered feature correlations to driver screening benchmarks (e.g., Web awareness pooled ρ=+0.31)
- **Model Variant Comparison**: Ensures Model A features are proper subset of Model B, identifying only autoregressive differences

### Evaluation Methods
- **Threshold Checks**: Hard limits (e.g., 163 stores, 52 weeks, 38k+ awareness rows)
- **Statistical Validation**: Mean/std/min/max checks, outlier detection
- **Smoothness Checks**: Week-to-week change constraints for seasonal curves
- **Coverage Analysis**: Percentage calculations for flags, mappings, and data availability

## Usage

### Running Individual Validation Scripts

#### Epic 1 & 2 Validation (Data + Seasonality)
```bash
python -m forecasting.regressor.validation.validate_epic1_epic2
```
**Runtime:** ~30-60 seconds
**Output:** Checkpoint-by-checkpoint validation with pass/warning/fail indicators

#### Epic 4 Validation (Feature Engineering)
```bash
python -m forecasting.regressor.validation.validate_epic4_pipeline
```
**Runtime:** 2-5 minutes (builds small canonical table samples)
**Output:** Comprehensive feature engineering pipeline validation report

#### Sister Store Mapping Validation
```bash
python -m forecasting.regressor.validation.validate_sister_store
```
**Runtime:** <10 seconds
**Output:** Sister-store mapping statistics and coverage validation

### Validation Output Interpretation

**Symbols:**
- ✓ (Checkmark): Check passed successfully
- ✗ (X-mark): Check failed, requires attention
- ⚠ (Warning): Potential issue detected but not critical, or informational

**Return Codes:**
- Exit code 0: All validations passed
- Exit code 1: One or more critical validations failed

## Expected Outcomes

### Successful Validation (validate_epic1_epic2.py)
```
VALIDATION COMPLETE
Total Checks: 45
Passed: 43 ✓
Warnings: 2 ⚠
Failed: 0 ✗
```

### Successful Validation (validate_epic4_pipeline.py)
```
OVERALL STATUS: ✓ EPIC 4 PIPELINE VALIDATION PASSED

The feature engineering pipeline is ready for:
  1. Full-scale canonical table generation (h=1-52)
  2. Integration with EPIC 5 (CatBoost model training)
  3. Production deployment (after final QA)
```

### Common Warning Scenarios
- Store count slightly different from 163 (stores opened/closed)
- Some awareness DMAs have <10 markets coverage
- CRM features have 10-30% NaN rates (expected for recent stores)
- Sister-store distances >1000 miles for remote locations

### Failure Scenarios Requiring Fix
- "NULL" strings not converted to NaN in traffic data
- Seasonal weights do not sum to 1.0 per DMA
- Leakage detected in feature computation
- Missing required columns in data loaders
- Model A contains features not in Model B (architecture violation)

## Integration with Broader Pipeline

The validation module sits between data ingestion/feature engineering and model training:

```
Data Sources → [Validation: Epic 1 & 2] → Clean Data + Seasonality
                                                ↓
                                    Canonical Training Table
                                                ↓
                                    [Validation: Epic 4] → Feature Coverage OK
                                                ↓
                                    Model Training (Epic 5)
                                                ↓
                                    Production Deployment
```

**When to Run:**
- **After data refresh**: Run `validate_epic1_epic2.py` to ensure data quality
- **After feature engineering changes**: Run `validate_epic4_pipeline.py` to verify no regressions
- **Before model training**: Run all validations to ensure data integrity
- **In CI/CD pipeline**: Include validation scripts as pre-training quality gates

## Validation Best Practices

1. **Always validate before training**: Never train models on unvalidated data
2. **Monitor warnings**: Even non-critical warnings can indicate data quality drift
3. **Rerun after data updates**: Validation is cheap compared to training; run frequently
4. **Document failures**: If validation fails, document root cause and fix before proceeding
5. **Use small samples first**: Epic 4 validation uses h=1-4 to catch issues early before full h=1-52 runs

## Related Documentation

- `../PROJECT_PLAN.md` - Complete engineering plan with all Epic checkpoints
- `../ENGINEERING_INSTRUCTIONS.md` - High-level model architecture and requirements
- `../features/validation.py` - Feature-level validation utilities used by Epic 4 script
- `../etl/canonical_table.py` - Canonical training table generation logic
- `../data_ingestion/` - Data loader implementations validated by these scripts

## Troubleshooting

**Issue:** Import errors when running validation scripts
**Solution:** Ensure you're running from project root with `python -m forecasting.regressor.validation.<script>`

**Issue:** Validation fails with "File not found"
**Solution:** Check that data files exist in expected locations (see `../paths.py` for configurations)

**Issue:** All checks show warnings about NaN rates
**Solution:** Some NaN rates are expected (e.g., CRM for new stores, awareness for unmapped DMAs). Review thresholds.

**Issue:** Sister-DMA validation shows "No DMAs require sister fallback"
**Solution:** This is valid if all DMAs have ≥3 years history. Validation passes automatically.

---

**Last Updated:** 2025-11-28
**Maintained By:** Forecasting Engineering Team
**Questions:** Refer to PROJECT_PLAN.md or escalate to model architecture team
