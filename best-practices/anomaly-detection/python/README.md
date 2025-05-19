# Anomaly Detection Best Practices with SAP HANA Machine Learning

This repository contains best practices for anomaly detection using SAP HANA Machine Learning (hana-ml) Python client library. The examples demonstrate how to detect outliers and anomalies in both general tabular data and time series data.

## Introduction

Anomaly detection is a critical task in many business scenarios, including fraud detection, system health monitoring, predictive maintenance, and quality control. This repository provides practical examples of implementing various anomaly detection algorithms with SAP HANA's Predictive Analysis Library (PAL) through the hana-ml Python client.

## Repository Structure

The repository is organized into three main directories:

- **[generate_anomaly_data/](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/anomaly-detection/python/generate_anomaly_data)**: Scripts for generating synthetic datasets with controlled anomalies
- **[general_anomaly/](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/anomaly-detection/python/general_anomaly)**: Jupyter notebooks implementing various anomaly detection algorithms for tabular data
- **[timeseries_anomaly/](https://github.com/SAP-samples/sap-btp-ai-best-practices/tree/main/best-practices/anomaly-detection/python/timeseries_anomaly)**: Jupyter notebooks implementing time series anomaly detection algorithms

## Prerequisites

- SAP HANA Cloud or on-premise instance with Predictive Analysis Library (PAL)
- Python 3.6+
- Required Python packages (see `requirements.txt`)

## Setup and Installation

1. Clone this repository.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Configure your connection settings to SAP HANA:
   - Create a `.env` file in the root directory with the following variables:
   ```
   hana_address=your_hana_host
   hana_port=your_hana_port
   hana_user=your_username
   hana_password=your_password
   hana_encrypt=true
   HANA_SCHEMA=your_schema_name
   ```

## Generating Synthetic Data

Two scripts are provided for generating synthetic data with controlled anomalies:

1. **[anomaly_generator.py](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/anomaly-detection/python/generate_anomaly_data/anomaly_generator.py)**: Generates tabular data with multiple features and controlled anomaly distributions
2. **[time_series_anomaly_generator.py](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/anomaly-detection/python/generate_anomaly_data/time_series_anomaly_generator.py)**: Generates time series data with seasonal patterns and injected point anomalies

Example usage:
```bash
cd generate_anomaly_data
python anomaly_generator.py
python time_series_anomaly_generator.py
```

## Anomaly Detection Algorithms

### General Anomaly Detection (Tabular Data)

The following notebooks demonstrate anomaly detection methods for tabular data:

1. **[isolation_forest.ipynb](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/anomaly-detection/python/general_anomaly/isolation_forest.ipynb)**: Uses Isolation Forest, an ensemble-based algorithm that efficiently isolates anomalies by randomly partitioning the data
   - Suitable for high-dimensional data
   - Works well with small sample sizes
   - No assumptions about data distribution

2. **[dbscan_anomaly.ipynb](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/anomaly-detection/python/general_anomaly/dbscan_anomaly.ipynb)**: Implements Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
   - Does not require specifying the number of clusters
   - Effective at finding clusters of arbitrary shapes
   - Labels points in low-density regions as outliers

3. **[OneClassSVM.ipynb](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/anomaly-detection/python/general_anomaly/OneClassSVM.ipynb)**: Implements One-Class Support Vector Machine
   - Works well with high-dimensional data
   - Suitable for cases with limited examples of anomalies
   - Effective at defining a tight boundary around normal data

4. **[outlier_detection_kmeans.ipynb](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/anomaly-detection/python/general_anomaly/outlier_detection_kmeans.ipynb)**: Uses K-means clustering distance-based approach for anomaly detection
   - Simple and computationally efficient
   - Identifies points far from their nearest cluster centroids as anomalies
   - Assumes spherical clusters

### Time Series Anomaly Detection

The following notebook demonstrates anomaly detection for time series data:

1. **[OutlierDetectionTS.ipynb](https://github.com/SAP-samples/sap-btp-ai-best-practices/blob/main/best-practices/anomaly-detection/python/timeseries_anomaly/OutlierDetectionTS.ipynb)**: Implements time series anomaly detection with multiple approaches:
   - Auto mode (PAL selects appropriate methods)
   - Median smoothing with IQR-based outlier detection
   - LOESS smoothing with MAD-based outlier detection
   - No smoothing with Isolation Forest for anomaly detection

## Implementation Details

Each notebook provides:
- Connection setup to SAP HANA
- Loading and preparation of data
- Model training and prediction
- Evaluation and visualization of results
- Performance metrics when ground truth is available

## Key Features of hana-ml Implementation

1. **In-database processing**: Computations are performed inside the SAP HANA database, reducing data movement and leveraging SAP HANA's speed
2. **Scalability**: Supports large datasets through SAP HANA's columnar storage and parallel processing
3. **Integration**: Seamlessly integrates with Python data science ecosystem (pandas, matplotlib, etc.)
4. **Productionization**: Models can be deployed directly within SAP HANA for real-time scoring

## Best Practices

### Data Preparation
- Ensure proper data cleaning and normalization before anomaly detection
- Handle missing values appropriately
- Consider dimensionality reduction for high-dimensional data

### Algorithm Selection
- Isolation Forest: Good general-purpose algorithm with efficiency for large datasets
- DBSCAN: Effective when anomalies form clusters in low-density regions
- One-Class SVM: Suitable when normal data is well-defined and compact
- K-means: Simple approach when clusters are expected to be spherical
- Time Series methods: Select based on presence of seasonality/trends and detection requirements

### Parameter Tuning
- Contamination (Isolation Forest): Set based on domain knowledge or estimate from data
- MinPts and Eps (DBSCAN): Critical for defining density; can be auto-determined by hana-ml
- Kernel and Nu (One-Class SVM): Affect boundary flexibility
- Smoothing and outlier methods (time series): Select based on time series characteristics

### Evaluation
- Use precision, recall, and F1-score when ground truth is available
- Visualize results to validate anomaly detection performance
- Consider business impact and context when setting thresholds


