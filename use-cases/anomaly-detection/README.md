# Pharmaceutical Sales Order Anomaly Detection

A comprehensive machine learning and AI-powered system for detecting anomalies in pharmaceutical sales orders. This solution combines Isolation Forest-based anomaly detection with LLM-driven binary classification to identify suspicious orders and provide explainability insights through SHAP analysis.

## Overview

This application provides end-to-end anomaly detection for pharmaceutical sales orders with three key stages:

1. **Data Preprocessing & Feature Engineering** - Transform raw sales order data into machine-learning-ready features
2. **Model Training & Prediction** - Train Isolation Forest models with customer stratification for robust anomaly detection
3. **AI-Enhanced Verification** - Leverage LLMs to validate borderline anomalies and reduce false positives
4. **Interactive Dashboard** - Explore results through a Streamlit-based analytics interface

## Architecture

```
Raw Sales Data (CSV)
         |
         v
Data Preprocessing (data_preprocessing.py)
- Load & consolidate duplicate lines
- Extract 12 anomaly signals
- Generate SHAP-ready features
         |
         v
Feature-Rich Dataset (merged_with_features.csv)
         |
         +----> Training Pipeline (training_pipeline.py)
         |      - Train Isolation Forest models
         |      - Optional customer stratification
         |      - SHAP explainability
         |      - Contamination rate tuning
         |
         +----> Batch Processing (batch_anomaly_processing.py)
                - Run ML anomaly detection
                - Flag borderline cases (score <= 0.05)
                - Generate SHAP explanations
                - Run AI classification for verification
                - Apply business rules
         |
         v
Enhanced Results with AI Verdict & Context Notes
         |
         v
Interactive Dashboard (__main__.py)
- Select & Explore Orders
- Monthly Trends & Metrics
- Model Fine-Tuning Controls
```

## Data Preprocessing

The `data_preprocessing.py` module transforms raw sales order data into anomaly detection-ready features.

### Input Data Requirements

**Input Format:**
- CSV files (`.csv` extension)
- Multiple CSV files can be placed in the data directory; they will be merged chronologically
- Dates should be in `MM/DD/YY` format (e.g., `01/15/24`)
- Entry time should be in `HH:MM:SS AM/PM` format (e.g., `09:30:45 AM`)
- Numeric columns may include thousands separators (commas); these are automatically removed

**Minimum Required Columns (10 columns):**
For basic anomaly detection functionality, the following columns are required:
1. `Sales Document Number` - Order identifier
2. `Sales Document Item` - Line item number
3. `Sold To number` - Customer ID
4. `Material Number` - Product ID
5. `Ship-To Party` - Delivery destination
6. `Sales Document Created Date` - Order creation date
7. `Sales Order item qty` - Ordered quantity
8. `Unit Price` - Price per unit
9. `Order item value` - Total line item value
10. `Sales unit` - Unit of measure

**Note:** If columns are missing, the preprocessing script will gracefully handle their absence, but certain features will be disabled. For example, without `Actual GI Date`, fulfillment time anomaly detection will be skipped.

**Example Input Data Shape:**
```
Sales Document Number | Sales Document Item | Sold To number | Material Number | Ship-To Party | Sales Document Created Date | Sales Order item qty | Unit Price | Order item value | Sales unit
--------------------|---------------------|----------------|-----------------|--------------|---------------------------|---------------------|------------|------------------|------------
123456              | 10                  | CUST001        | MAT001          | SHIP001      | 01/15/24                  | 100                 | 25.50      | 2550.00          | EA
123456              | 20                  | CUST001        | MAT002          | SHIP001      | 01/15/24                  | 50                  | 12.75      | 637.50           | CS
...
```

### Expected Input Columns

The preprocessing script handles missing columns gracefully. Columns are categorized as **Required** (core functionality) or **Optional** (enhanced features):

#### Required Columns (Core Functionality)

These columns are essential for the anomaly detection pipeline to function:

**Identifiers:**
- `Sales Document Number` - **Required** - Unique order identifier (used for grouping and deduplication)
- `Sales Document Item` - **Required** - Line item number within the order
- `Sold To number` - **Required** - Customer ID (used for customer-level anomaly detection)
- `Material Number` - **Required** - Product ID (used for material-level features)
- `Ship-To Party` - **Required** - Delivery destination (used for destination anomaly detection)

**Dates & Times:**
- `Sales Document Created Date` - **Required** - Order creation timestamp (format: `MM/DD/YY`). Used for temporal features, duplicate detection, and monthly volume analysis

**Quantity & Pricing:**
- `Sales Order item qty` - **Required** - Ordered quantity (used for quantity deviation features)
- `Unit Price` - **Required** - Price per unit (used for pricing anomaly detection)
- `Order item value` - **Required** - Total line item value (used for value validation and filtering)

**Material & Unit:**
- `Sales unit` - **Required** - Unit of measure (EA, CS, BOX, etc.) - Used for UoM anomaly detection

#### Optional Columns (Enhanced Features)

These columns enable additional anomaly detection features but are not strictly required:

**Identifiers:**
- `Customer PO number` - Customer's purchase order number (for reference)

**Dates & Times:**
- `Entry time` - Order entry time (format: `HH:MM:SS AM/PM`) - Used for temporal analysis
- `Requested delivery date` - Customer-requested delivery date
- `Actual GI Date` - **Recommended** - Goods issue (shipment) date (enables fulfillment time anomaly detection)
- `Invoice Creation Date` - Invoice date
- `Shelf Life Expiration Date` - Product expiration date
- `Original Requested Delivery Date` - Original delivery request
- `Header Pricing Date` - Pricing reference date
- `Item Pricing Date` - Item pricing date
- `Batch Manufacture Date` - Manufacturing date

**Quantity & Fulfillment:**
- `Actual quantity delivered` - Delivered quantity
- `Actual quantity delivered (ELC only)` - Delivered (emergency fulfillment only)
- `Quantity invoiced` - Invoiced quantity
- `Confirmed Quantity` - Confirmed quantity
- `Quanty open to ship (ELC)` - Outstanding quantity

**Pricing & Value:**
- `Invoiced value` - Value invoiced
- `Value Open` - Outstanding value
- `Subtotal 1`, `Subtotal 2`, `Subtotal 3`, `Subtotal 4`, `Subtotal 5`, `Subtotal 6` - Pricing subtotals

**Material & Status:**
- `Material Description` - **Recommended** - Product description (used in rare material detection explanations)
- `BillingStatus Desc` - Order status (e.g., "Cancelled", "Normal"). If present, cancelled orders are automatically filtered out

### Column Usage Matrix

| Feature | Required Columns | Optional Columns Used |
|---------|-----------------|----------------------|
| First-time order detection | `Sold To number`, `Material Number`, `Sales Document Created Date` | - |
| Rare material detection | `Material Number` | `Material Description` (for explanations) |
| Quantity deviation | `Sold To number`, `Material Number`, `Sales Order item qty` | - |
| Unusual UoM | `Sold To number`, `Material Number`, `Sales unit` | - |
| Duplicate detection | `Sold To number`, `Material Number`, `Sales Order item qty`, `Sales Document Created Date` | - |
| Monthly volume | `Sold To number`, `Material Number`, `Sales Order item qty`, `Sales Document Created Date` | - |
| Delivery destination | `Sold To number`, `Ship-To Party` | - |
| Pricing anomalies | `Material Number`, `Sales unit`, `Unit Price`, `Sold To number` | - |
| Value mismatch | `Unit Price`, `Sales Order item qty`, `Order item value` | - |
| Fulfillment time | `Material Number`, `Ship-To Party`, `Sales Document Created Date` | `Actual GI Date` |

### Processing Steps

#### 1. Data Consolidation
```python
# Merges multiple CSV files from the data directory
# Handles cancelled orders and negative values filtering
# Consolidates duplicate material lines within order/ship-to
```

**Key Filters:**
- Removes orders with `BillingStatus Desc = 'Cancelled'`
- Removes lines with `Order item value < 0`
- Consolidates multiple lines for same (Order, Material, Ship-To, UoM) using first-occurrence rule

#### 2. Feature Engineering (12 Anomaly Signals)

**A. First-Time Order Detection**
- Feature: `is_first_time_cust_material_order`
- Identifies orders for a (Customer, Material) pair never ordered before
- Useful for detecting new product exploration

**B. Rare Material Detection**
- Feature: `is_rare_material`
- Flags materials appearing less than 3 times in historical data
- Indicates unusual product ordering patterns

**C. Quantity Deviation Analysis**
- Features: `qty_z_score`, `qty_deviation_from_mean`, `is_qty_outside_typical_range`
- **Robust multi-regime statistics** (minimum history gates):
  - **Large history** (n >= 30): Mean/std + percentile bands (5th-95th)
  - **Medium history** (10 <= n < 30): Median/MAD robust z-score + IQR whiskers
  - **Small history** (n < 10): Neutral defaults (z=0, flag=False)
- Reduces false positives on sparse customer-material pairs

**D. Unusual Unit of Measure**
- Feature: `is_unusual_uom`
- Compares current order's UoM against customer's historical mode
- Flags when UoM differs AND sufficient history exists (>= 10 samples)

**E. Suspected Duplicate Orders**
- Feature: `is_suspected_duplicate_order`
- Detects orders within 24 hours with identical (Customer, Material, Qty)
- Useful for catching accidental duplicates

**F. Quantity Trend Analysis**
- Feature: `qty_trend_slope_lastN`
- Calculates rolling slope over last 5 orders per (Customer, Material)
- Captures local quantity trends without requiring long history

**G. Monthly Volume Anomalies**
- Features: `month_rolling_z`, `order_share_of_month`, `is_order_qty_high_z`
- Computes rolling baseline (6-month window, excluding current month)
- Flags orders significantly above/below historical monthly volumes

**H. Unusual Delivery Destination**
- Feature: `is_unusual_ship_to_for_sold_to`
- Identifies rare ship-to locations (< 1% of customer's orders)
- Requires sufficient customer order history (>= 50 orders)

**I. Pricing Anomalies - Unit Price**
- Features: `is_unusual_unit_price`, `price_z_vs_customer`
- **Material-level pricing**: IQR bands (small n) or percentile bands (large n)
- **Customer-level pricing**: MAD-based robust z-score
- Compares individual order price against (Material, UoM) and (Customer, Material) benchmarks

**J. Pricing Anomalies - Value Mismatch**
- Feature: `is_value_mismatch_price_qty`
- Validates: `Order item value ≈ Unit Price × Qty` (tolerance: 1%)
- Catches data entry errors or billing inconsistencies

**K. Fulfillment Time Anomalies**
- Features: `fulfillment_duration_days`, `is_unusual_fulfillment_time`
- Calculates: Days from order creation to goods issue
- Flags when fulfillment deviates from (Material, Ship-To) baseline
- Similar robust multi-regime logic as quantity deviations

**L. Anomaly Explanations**
- Feature: `anomaly_explanation`
- Human-readable summary of triggered anomaly signals
- Used for dashboard display and AI context

### Output Columns

The preprocessed dataset includes all original columns plus 40+ derived features organized by anomaly type. See `export_selected_features()` for the exact column ordering used in downstream ML pipelines.

### Configuration Constants

Located at top of `data_preprocessing.py`:

```python
MIN_QTY_HISTORY_ROBUST = 10      # Threshold for robust statistics
MIN_QTY_HISTORY_STRICT = 30      # Threshold for mean/std
MIN_PRICE_SAMPLES_ROBUST = 20    # Price history gate
MIN_UOM_SAMPLES = 10             # UoM history gate
MIN_SHIPTO_ORDERS = 50           # Delivery destination gate
MIN_FULFILL_SAMPLES_ROBUST = 20  # Fulfillment history gate
TREND_WINDOW_ORDERS = 5          # Rolling window for trend
COMPARISON_EPS = 1e-6            # Floating-point tolerance
```

### Example Usage

**Basic preprocessing:**
```bash
python data_preprocessing.py
```

**Custom data directory:**
Modify the `data_directory` parameter in `main()` function (default: `'data'`):
```python
df = load_and_preprocess_data('/path/to/your/csv/files')
```

**Note:** The `main()` function currently hardcodes a data path. To use a custom directory, either:
1. Modify the `data_directory` parameter in `main()` function, or
2. Call `load_and_preprocess_data()` directly with your path:
```python
from data_preprocessing import load_and_preprocess_data
df = load_and_preprocess_data('/path/to/your/csv/files')
# Then proceed with feature engineering steps...
```

**Outputs:**
- `data/merged_with_features.csv` - Full feature set with all original columns plus 40+ derived features
- `merged_with_features_selected_ordered.csv` - Selected columns for ML modeling (ordered as expected by training pipeline)

**Processing Summary:**
- Merges all CSV files in the specified directory
- Filters out cancelled orders (if `BillingStatus Desc` column exists)
- Removes rows with negative `Order item value`
- Consolidates duplicate material lines within the same order/ship-to/UoM combination
- Generates 12 types of anomaly signals (40+ features)
- Produces human-readable anomaly explanations

---

## Training Pipeline

The `training_pipeline.py` module orchestrates model training with configurable backends, customer stratification, and explainability.

### Architecture

**Supported Backends:**
- **scikit-learn (sklearn)** - Isolation Forest (recommended)
- **SAP HANA ML** - HANA PAL Isolation Forest (enterprise backend)

**Model Configurations:**
- Standard single-model approach
- Customer-stratified approach (tier-based)

### Key Features

#### 1. Isolation Forest Model
- Anomaly detection algorithm: Isolation Forest
- Configurable contamination rate (default: auto)
- Supports contamination tuning: `--contamination 0.05` or `--contamination auto`
- Model hyperparameters:
  - `n_estimators`: Number of trees (default: 150, tunable)
  - `max_samples`: Samples per tree (default: 512, tunable)

#### 2. Customer Stratification
- **Motivation**: Segment customers by ordering volume for tier-specific models
- **Tiers**:
  - **Global**: Newly created sales orders
  - **Medium**: Medium-volume customers
  - **Large**: High-volume customers
- **Benefit**: Reduced false positives by fitting models to tier-specific distributions
- **Activation**: Pass `--customer-stratified` flag

#### 3. SHAP Explainability
- Generates TreeExplainer-based feature importance
- Shows which features contributed most to each anomaly prediction
- Activation: Pass `--shap` flag

#### 4. Feature Selection
- Supports UI-based feature selection via `SELECTED_FEATURES` environment variable
- Falls back to auto-detected features if not provided
- Validates feature existence before training

#### 5. Model Persistence
- Saves trained models to `results/` directory
- Can reload models with `--load-models` flag for consistent predictions
- Metadata includes feature names, contamination rate, and training info

### Usage

**Basic Training (sklearn backend, auto contamination):**
```bash
python training_pipeline.py --backend sklearn
```

**With Customer Stratification:**
```bash
python training_pipeline.py --backend sklearn --customer-stratified
```

**With SHAP Explanations:**
```bash
python training_pipeline.py --backend sklearn --shap --n-estimators 200
```

**Load & Reuse Saved Models:**
```bash
python training_pipeline.py --backend sklearn --load-models
```

**Via Environment Variables (UI-driven):**
```bash
export SELECTED_FEATURES="qty_z_score,price_z_vs_customer,month_rolling_z"
export N_ESTIMATORS=250
export MAX_SAMPLES=1024
python training_pipeline.py --backend sklearn
```

### Command Line Arguments Reference

The `training_pipeline.py` script supports the following command line arguments:

#### `--backend {hana|sklearn}`
- **Default**: `hana`
- **Options**: 
  - `sklearn` - scikit-learn Isolation Forest (recommended, recommended, fast)
  - `hana` - SAP HANA ML Isolation Forest (enterprise backend, requires hana-ml package)
- **Description**: Selects the machine learning backend for training
- **Example**: `--backend sklearn`

#### `--contamination {auto|float}`
- **Default**: `auto`
- **Options**:
  - `auto` - Automatically estimate contamination rate from data
  - Any float value between 0.0 and 1.0 (e.g., `0.05` for 5%, `0.1` for 10%)
- **Description**: Sets the contamination rate (fraction of anomalies expected in dataset)
- **Example**: `--contamination 0.045`

#### `--shap`
- **Default**: Not enabled (flag)
- **Options**: Present or absent
- **Description**: Enables SHAP TreeExplainer for feature importance analysis. Generates:
  - Global summary plots showing average feature importance
  - Dependence plots for top features
  - Per-sample SHAP value explanations
- **Performance**: Adds ~30-50% to training time but provides detailed explanations
- **Example**: `--shap`

#### `--customer-stratified`
- **Default**: Not enabled (flag)
- **Options**: Present or absent
- **Description**: Trains separate Isolation Forest models for different customer tiers:
  - **Global tier**: New/small customers with few orders
  - **Medium tier**: Medium-volume customers
  - **Large tier**: High-volume customers
- **Benefit**: Reduces false positives by fitting models to tier-specific distributions
- **Example**: `--customer-stratified`

#### `--load-models`
- **Default**: Not enabled (flag)
- **Options**: Present or absent
- **Description**: Loads previously trained models instead of training new ones
- **Speed**: Much faster for prediction-only scenarios
- **Compatibility**: Models must be compatible in terms of features and contamination rate
- **Example**: `--load-models`

#### `--file <path>`
- **Default**: `None` (uses default dataset from settings)
- **Type**: File path (string)
- **Description**: Path to custom CSV dataset file to use instead of default
- **Format**: Must be preprocessed CSV with feature columns
- **Example**: `--file /path/to/custom_data.csv`

#### `--n-estimators <int>`
- **Default**: `None` (falls back to env var `N_ESTIMATORS` or 150)
- **Type**: Integer
- **Range**: Typically 50-500 (higher = more trees = better accuracy but slower)
- **Description**: Number of isolation trees to train
- **Recommendation**: 150-200 for balanced accuracy/speed, 300-400 for high accuracy
- **Example**: `--n-estimators 250`

#### `--max-samples {int|auto}`
- **Default**: `None` (falls back to env var `MAX_SAMPLES` or 512)
- **Type**: Integer or string `"auto"`
- **Options**:
  - `auto` - Use all samples
  - Integer (e.g., 512, 1024, 2048) - Number of samples per tree
- **Description**: Maximum number of samples drawn to train each tree
- **Recommendation**: 256-1024 for balanced training, `auto` for full data access
- **Example**: `--max-samples 1024` or `--max-samples auto`

### Real-World Usage Examples

#### Example 1: Quick Baseline (30 seconds)
```bash
python training_pipeline.py --backend sklearn
```
- Uses default dataset
- Auto contamination detection
- Single model (no stratification)
- No SHAP (fast)
- Default hyperparameters (150 trees, 512 samples per tree)

#### Example 2: Production Model with Explainability (3-5 minutes)
```bash
python training_pipeline.py --backend sklearn --shap --customer-stratified
```
- Trains stratified models for each customer tier
- Generates SHAP explanations for all samples
- Suitable for detailed analysis and auditing
- Results include summary plots and feature importance

#### Example 3: High-Accuracy Model (5-10 minutes)
```bash
python training_pipeline.py --backend sklearn --n-estimators 400 --max-samples auto --contamination 0.05
```
- More trees (400 vs default 150) for better accuracy
- Uses all samples per tree (auto)
- Fixed 5% contamination rate
- Higher accuracy at the cost of training time

#### Example 4: Fast Prediction with Loaded Models (10-30 seconds)
```bash
python training_pipeline.py --backend sklearn --load-models
```
- Loads previously saved models
- Skips training entirely
- Performs prediction on test set
- Much faster for repeated scoring

#### Example 5: Stratified with SHAP and Custom Data
```bash
python training_pipeline.py \
  --backend sklearn \
  --customer-stratified \
  --shap \
  --file /path/to/my_dataset.csv \
  --contamination 0.03
```
- Uses custom dataset
- Stratifies by customer tier
- Generates SHAP explanations
- 3% contamination rate

#### Example 6: Load Stratified Models with SHAP
```bash
python training_pipeline.py \
  --backend sklearn \
  --customer-stratified \
  --load-models \
  --shap
```
- Loads previously trained stratified models
- Generates SHAP explanations on test set
- Useful for explaining predictions without retraining

#### Example 7: Hyperparameter Optimization (10-15 minutes)
```bash
python training_pipeline.py \
  --backend sklearn \
  --n-estimators 500 \
  --max-samples 2048 \
  --customer-stratified \
  --shap \
  --contamination 0.045
```
- Large number of trees (500)
- More samples per tree (2048)
- Stratified models
- Full SHAP analysis
- Best accuracy for final deployment

#### Example 8: Environment Variable Configuration (for UI)
```bash
export SELECTED_FEATURES="qty_z_score,price_z_vs_customer,month_rolling_z,is_qty_outside_typical_range,is_unusual_unit_price"
export N_ESTIMATORS=200
export MAX_SAMPLES=1024
python training_pipeline.py --backend sklearn --customer-stratified --shap
```
- Features selected via environment variable
- Hyperparameters configured via environment variables
- Ideal for UI-driven workflows
- Command line flags override environment variables if both set

#### Example 9: HANA Backend (if available)
```bash
python training_pipeline.py \
  --backend hana \
  --contamination 0.05 \
  --customer-stratified
```
- Uses SAP HANA ML backend (requires hana-ml package)
- Falls back to sklearn if HANA not available
- Suitable for enterprise environments with HANA infrastructure

#### Example 10: Multiple Training Runs for Comparison
```bash
# Run 1: Baseline
python training_pipeline.py --backend sklearn

# Run 2: With stratification
python training_pipeline.py --backend sklearn --customer-stratified

# Run 3: With SHAP
python training_pipeline.py --backend sklearn --shap

# Run 4: Combined
python training_pipeline.py --backend sklearn --customer-stratified --shap
```
- Compare results across different configurations
- Each creates a separate results directory with timestamp
- Use dashboard to compare metrics

#### Example 11: Custom Dataset + Hyperparameter Tuning
```bash
python training_pipeline.py \
  --backend sklearn \
  --file datasets/all_data.csv \
  --n-estimators 300 \
  --max-samples 1500 \
  --contamination auto \
  --shap
```
- Processes custom dataset
- Balanced hyperparameters
- Auto-detected contamination
- Detailed SHAP analysis

#### Example 12: Production Batch Scoring (Very Fast)
```bash
python training_pipeline.py \
  --backend sklearn \
  --load-models \
  --customer-stratified
```
- Loads pre-trained models
- No retraining needed
- Fast prediction on new data
- Perfect for batch processing pipelines

### Environment Variable Integration

The training pipeline respects the following environment variables (command line args take priority):

```bash
# Feature Selection (UI-driven)
export SELECTED_FEATURES="feat1,feat2,feat3"

# Model Hyperparameters
export N_ESTIMATORS=200
export MAX_SAMPLES=1024

# Contamination Rate
export CONTAMINATION_RATE=0.045

# Backend Selection
export ML_BACKEND=sklearn
```

**Priority Order** (highest to lowest):
1. Command line arguments (`--arg value`)
2. Environment variables (`export VAR=value`)
3. Configuration file defaults (`config/settings.py`)
4. Hardcoded defaults in code

### Output Files & Directory Structure

Each training run creates a timestamped results directory with this structure:

```
results/anomaly_detection_results_backend_{backend}_contamination_{rate}[_customer_stratified][_shap]/
├── models/
│   ├── sklearn_model.joblib                    # Single model (if no stratification)
│   ├── stratified_model_global.joblib          # Global tier model (if stratified)
│   ├── stratified_model_medium.joblib          # Medium tier model (if stratified)
│   ├── stratified_model_large.joblib           # Large tier model (if stratified)
│   ├── customer_tiers.json                     # Customer assignments to tiers (if stratified)
│   └── model_metadata.json                     # Feature names, contamination, training date
├── merged_with_features_selected_ordered.csv   # Preprocessed data with all features
├── predictions_test.csv                        # Predictions on test set
├── shap_global_summary.png                     # SHAP summary plot (if --shap)
├── shap_dependence_plots/                      # Individual feature plots (if --shap)
│   ├── shap_dependence_qty_z_score.png
│   ├── shap_dependence_price_z_vs_customer.png
│   └── ...
├── feature_importance.csv                      # Feature importance ranking
├── summary_report.txt                          # Human-readable summary
└── metrics.json                                # Performance metrics

```

### Monitoring Training Progress

The script outputs detailed progress information:

```
================================================================================
PHARMACEUTICAL ANOMALY DETECTION WITH ISOLATION FOREST
================================================================================
Start time: 2025-10-22 14:30:00
Backend selected: SKLEARN
Contamination mode: auto
SHAP explanations: Enabled
Customer stratification: Enabled
Load saved models: Disabled
Using backend: SKLEARN

Results will be saved to: results/anomaly_detection_results_backend_sklearn_contamination_auto_customer_stratified_shap/

Data loading...
Loaded 50000 records from default dataset
Train/test split: 40000 / 10000

USING UI-SELECTED FEATURES
================================================================================
Features from UI: 15
 1. qty_z_score
 2. price_z_vs_customer
 3. month_rolling_z
...

Using model parameters:
  n_estimators: 200
  max_samples: 1024
  (Parameters from command line arguments)

Training new stratified models...
  - Training global tier model...
  - Training medium tier model...
  - Training large tier model...

Generating SHAP explanations...
[###########          ] 50% complete

SAVING TRAINED MODELS
================================================================================
Models saved successfully!
To reuse these models, run with --load-models flag

ANALYSIS COMPLETE
================================================================================
End time: 2025-10-22 14:35:42
Backend used: sklearn (customer-stratified, stratified)
Results saved to: results/anomaly_detection_results_backend_sklearn_contamination_auto_customer_stratified_shap/
```

### Output Artifacts

**Model Directory Structure:**
```
results/anomaly_detection_results_backend_sklearn_contamination_auto_shap/
├── models/
│   ├── sklearn_model.joblib              # Trained model
│   ├── stratified_model_global.joblib    # (if --customer-stratified)
│   ├── stratified_model_medium.joblib    # (if --customer-stratified)
│   ├── stratified_model_large.joblib     # (if --customer-stratified)
│   ├── customer_tiers.json               # (if --customer-stratified)
│   └── model_metadata.json               # Feature names, contamination, date
├── merged_with_features_selected_ordered.csv  # Preprocessed data
├── shap_global_summary.png               # (if --shap)
├── shap_dependence_plots/                # (if --shap)
└── summary_report.txt                    # Training summary
```

---

## Batch Anomaly Processing & AI Verification

The `batch_anomaly_processing.py` module performs production-grade anomaly detection with AI-enhanced verification for borderline cases.

### Three-Stage Verification Pipeline

#### Stage 1: ML Anomaly Detection
```python
# Run trained Isolation Forest models
anomaly_scores, anomaly_labels, model_assignments = stratified_model.predict(...)
```

**Outputs:**
- `anomaly_score` [0, 1]: Normalized anomaly score (0 = normal, 1 = anomalous)
- `predicted_anomaly` [0, 1]: Binary prediction from model
- `model_used`: Which tier's model made the prediction (global/medium/large)

**Decision Boundary:**
- `anomaly_score <= 0`: ML-detected anomaly
- `0 < anomaly_score <= 0.05`: Borderline case (triggers extended analysis)

#### Stage 2: SHAP Explainability
```python
# For all orders with anomaly_score <= 0.05 (ML anomalies + borderline cases)
shap_explanations = create_shap_explanations(
    model=model,
    X_train=training_sample,
    X_test=anomalous_orders,
    feature_columns=features
)
```

**Features Generated:**
- Top 5-10 contributing features per order
- Feature contribution values and directions
- Localized SHAP summary plots

**Use Case**: Provides explainability for both:
- ML-detected anomalies (why is this order suspicious?)
- Borderline cases (why is this close to the boundary?)

#### Stage 3: AI Binary Classification (LLM Verification)
```python
# For extended analysis cases (anomaly_score <= 0.05)
ai_result = generate_ai_binary_classification_with_images(
    row=order_data,
    features_df=all_orders,
    image_paths=shap_visualizations
)
```

**AI Classification Workflow:**

1. **Input Context** (passed to LLM):
   - Order details: Customer, Material, Quantity, Price, Dates
   - Anomaly signals: Which features are unusual
   - SHAP explanations: Why the model flagged this order
   - Historical context: Customer's typical patterns
   - Feature visualizations: Charts showing deviation from baseline

2. **LLM Binary Decision**:
   - Output: "True" (anomaly confirmed) or "False" (false positive)
   - Rationale: Detailed explanation of the decision

3. **Context Notes** (on "False" decision):
   - Extract and store brief context explaining why it's normal
   - Forward to next step for escalation summary
   - Useful for model refinement feedback

**Key Advantage**: Reduces false positives while retaining sensitivity to real anomalies

### Processing Flow

```python
# Load preprocessed data
data = load_and_filter_data(DATA_FILE, days=None)

# Stage 1: ML anomaly detection
data = run_ml_anomaly_detection(data, models, customer_tiers, feature_columns)
#   Adds: anomaly_score, predicted_anomaly, model_used, extended_analysis

# Stage 2: SHAP explanations for extended analysis (score <= 0.05)
data = generate_shap_explanations_for_extended_analysis(data, models, feature_columns)
#   Adds: shap_explanation

# Stage 3: AI classification for extended analysis cases
data = run_ai_classification_for_extended_analysis(data)
#   Adds: ai_anomaly_result

# Business rules & summary
data = apply_business_rules(data)
#   Adds: Blocked by Business Rules, Business Rules columns

# Metrics & reporting
metrics = calculate_metrics(data)
save_results(data, metrics, OUTPUT_DIR)
```

### Output Columns

**Result CSV includes:**
- Order identifiers & dates
- Quantity, pricing, fulfillment data
- `anomaly_score`: ML score [0, 1]
- `predicted_anomaly`: ML binary prediction
- `extended_analysis`: True if score <= 0.05
- `ai_anomaly_result`: AI binary decision (True/False/None)
- `shap_explanation`: Feature importance summary
- `Blocked by Business Rules`: Boolean flag
- `Business Rules`: Triggered rule details

### Usage

```bash
python batch_anomaly_processing.py
```

Outputs:
- `anomaly_detection_results_YYYYMMDD_HHMMSS.csv` - Enhanced data with AI verdicts
- `metrics_YYYYMMDD_HHMMSS.json` - Summary metrics

---

## Dashboard & Interactive Exploration

The `__main__.py` module provides a Streamlit-based interactive dashboard for exploring anomalies and managing the ML pipeline.

### Pages

#### 1. Select a Sales Order
**Purpose**: Deep-dive investigation of individual orders

**Features:**
- Search & filter orders by Sales Document Number, Customer, Material
- View complete order details with all 40+ features
- Display anomaly signals triggered for the order
- Show SHAP explanation breakdown (top contributing features)
- See AI classification verdict with reasoning
- Historical context: Customer's typical patterns
- Related orders: Similar orders from same customer/material

#### 2. Monthly Dashboard
**Purpose**: High-level trend analysis and cohort performance

**Features:**
- Calendar heatmap of anomaly rates by date
- Daily order counts and anomaly contamination rates
- Trend lines for ML vs AI-detected anomalies
- Tier-specific metrics (global/medium/large customers)
- Monthly comparison: This month vs prior months
- Material & customer anomaly rankings
- Drill-down capability to daily/hourly breakdowns

#### 3. Fine Tuning
**Purpose**: Model management and retraining

**Features:**
- View current model metadata (features, contamination, training date)
- Update feature selection and model hyperparameters
- Retrain models with new settings
- Compare model performance metrics
- Load/save alternative model versions
- Monitor training progress and diagnostics

### Sample Data

The `datasets/` directory contains example preprocessed data:

```
datasets/
├── all_data.csv (14MB, ~50K orders)
└── (raw CSV with all 40+ features from data_preprocessing.py)
```

**Sample columns in all_data.csv:**
```
Sales Document Number | Sales Document Item | Material Number | Sold To number
Sales Order item qty  | Unit Price          | Order item value | Sales Document Created Date
is_first_time_cust_material_order | is_rare_material | qty_z_score | is_qty_outside_typical_range
is_unusual_uom | is_suspected_duplicate_order | month_rolling_z | is_unusual_ship_to_for_sold_to
is_unusual_unit_price | is_value_mismatch_price_qty | is_unusual_fulfillment_time
anomaly_explanation | [+ 20+ more feature columns]
```

### Dashboard Usage

**Start the application:**
```bash
streamlit run __main__.py
```

**Access the dashboard:**
- Open browser to `http://localhost:8501`
- Use sidebar to navigate between pages
- Query parameters support deep-linking: `?tab=1_Select_a_Sales_Order`

**Workflow Example:**
1. Start at **Monthly Dashboard** to identify high-anomaly days
2. Drill into **Select a Sales Order** for specific orders
3. Review AI verdicts and SHAP explanations
4. Return to **Fine Tuning** to adjust model sensitivity if needed

---

## End-to-End Workflow Example

### Week 1: Initial Setup

```bash
# 1. Preprocess raw sales data
python data_preprocessing.py
# Output: merged_with_features.csv

# 2. Train baseline model with SHAP
python training_pipeline.py --backend sklearn --shap --customer-stratified
# Output: results/anomaly_detection_results_.../models/

# 3. View dashboard
streamlit run __main__.py
# Explore monthly trends and sample orders
```

### Week 2: Production Batch Processing

```bash
# 1. Run batch anomaly detection on two weeks of new orders
export DAYS_TO_PROCESS=14
python batch_anomaly_processing.py
# Output: anomaly_detection_results_YYYYMMDD_HHMMSS.csv

# 2. Review results in dashboard
# Navigate to "Select a Sales Order" page
# Search for orders with ai_anomaly_result=True
# Analyze root causes and patterns
```

### Week 3: Model Refinement

```bash
# 1. In dashboard, go to "Fine Tuning" page
# 2. Select key discriminative features (e.g., qty_z_score, price_z_vs_customer)
# 3. Retrain with optimized hyperparameters
# 4. Compare new model metrics vs baseline

# 5. Save improved model
python training_pipeline.py --backend sklearn --load-models
# Re-scores all data with new model
```

---

## Technical Requirements

### Dependencies

```
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 1.0.0
streamlit >= 1.10.0
matplotlib >= 3.4.0
shap >= 0.41.0
plotly >= 5.0.0
joblib >= 1.0.0
```

### Optional Dependencies

```
hana-ml              # For SAP HANA ML backend
openai               # For LLM-based AI verification
anthropic            # Alternative LLM provider
```

### Data Requirements

- Minimum dataset size: 1,000 orders
- Recommended: 10,000+ orders for robust customer tiers
- Input CSV with columns specified in "Expected Input Columns" section

### Performance Notes

- Preprocessing 50K orders: ~30 seconds
- Training Isolation Forest: ~10 seconds (sklearn) or ~60 seconds (HANA)
- Batch prediction: ~100 orders/second
- SHAP explanation generation: ~5 seconds per 100 orders
- AI classification (LLM calls): ~2-5 seconds per order (network-dependent)

---

## Troubleshooting

**Q: "No CSV files found in the specified directory"**
- Ensure data files are in the configured directory
- Check file naming: must end with `.csv`

**Q: "Insufficient data for train/test split"**
- Minimum 100 orders required
- Check for data filtering removing too many rows

**Q: SHAP explanations take too long**
- Reduce dataset size or sample for SHAP generation
- Use SHAP with TreeExplainer (faster than LIME)

**Q: AI classification returns None**
- Check LLM API keys and connectivity
- Review LLM error logs in console
- Ensure order data contains required columns

**Q: Dashboard shows "No data loaded"**
- Verify data path in `app_setup.py`
- Ensure CSV file exists and is readable
- Check column names match expected schema

---

## File Structure

```
use-cases/anomaly-detection/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── data_preprocessing.py                  # Feature engineering
├── training_pipeline.py                   # Model training orchestration
├── batch_anomaly_processing.py            # Production batch scoring + AI verification
├── business_rules.py                      # Domain-specific checks
├── __main__.py                            # Streamlit dashboard entry point
├── datasets/
│   └── all_data.csv                      # Example preprocessed data (14MB)
├── config/
│   ├── cli.py                            # Command-line argument parsing
│   ├── settings.py                       # Global configuration
│   └── ...
├── data/
│   ├── loader.py                         # Data loading utilities
│   ├── features.py                       # Feature selection & preparation
│   └── ...
├── models/
│   ├── sklearn_model.py                  # Sklearn Isolation Forest wrapper
│   ├── hana_model.py                     # HANA ML wrapper
│   ├── stratified_model.py               # Customer tier stratification
│   ├── persistence.py                    # Model save/load logic
│   └── ...
├── explainability/
│   ├── shap_explainer.py                 # SHAP TreeExplainer integration
│   ├── ai_explanation_generator.py       # LLM-based verification
│   └── ...
├── pages/
│   ├── 1_Select_a_Sales_Order.py         # Order drill-down page
│   ├── 2_Monthly_Dashboard.py            # Trend analysis page
│   └── 3_Fine_Tuning.py                  # Model management page
├── visualization/
│   ├── feature_analysis.py               # Feature importance charts
│   └── ...
├── services/
│   ├── anomaly_service.py                # ML anomaly detection service
│   └── ...
├── utils/
│   ├── state.py                          # Streamlit session state management
│   ├── directories.py                    # File system utilities
│   └── ...
├── evaluation/
│   └── metrics.py                        # Evaluation metrics computation
├── reporting/
│   └── summary.py                        # Summary report generation
├── results/                              # Output directory (created at runtime)
│   └── anomaly_detection_results_.../
└── batch_results/                        # Batch processing outputs
```

---

## References

- **Isolation Forest**: Liu et al., 2008. "Isolation Forest"
- **SHAP**: Lundberg & Lee, 2017. "A Unified Approach to Interpreting Model Predictions"
- **Robust Statistics**: Huber, 1981. "Robust Statistics"
- **Customer Segmentation**: RFM (Recency, Frequency, Monetary) analysis

---

## Support & Contributing

For issues, questions, or improvements:
1. Review the "Troubleshooting" section above
2. Check the configuration files for environment-specific settings
3. Examine console output for detailed error messages
4. Consult individual module docstrings for function-level documentation

---

**Last Updated**: October 2025
**Version**: 1.0.0
