# SAP BTP AI Best Practices – Narrow AI

This section of the repository contains best practices for implementing Narrow AI solutions on the SAP Business Technology Platform (BTP). Each subdirectory focuses on a specific predictive or classical machine learning technique, with sample code, documentation, and guidance.

## Contents

| Folder | Technique | Implementation |
|---|---|---|
| [anomaly-detection](./anomaly-detection/README.md) | Outlier detection for tabular, time-series and regression data | Python, SAP HANA Predictive Analysis Library (PAL) via hana-ml |
| [classification](./classification/README.md) | Supervised classification | Python, PAL via hana-ml |
| [clustering](./clustering/README.md) | Unsupervised clustering (KMeans) | Python, PAL via hana-ml |
| [regression](./regression/README.md) | Linear regression | Python, PAL via hana-ml |
| [time-series-forecasting](./time-series-forecasting/README.md) | Forecasting | Python, PAL via hana-ml |
| [RPT-1](./RPT-1/README.md) | Classification and regression on tabular data with SAP-RPT, a pretrained relational foundation model | Python, TypeScript, Java (SAP Cloud SDK for AI) |

### PAL or SAP-RPT?

- Use **PAL** when your data is in SAP HANA Cloud and you want to train, tune and explain a model for one well-defined task, keeping the computation close to the data.
- Use **SAP-RPT** when you want predictions without task-specific training: you send example rows as context together with the rows to predict, and the model infers the target values through in-context learning. This is useful for many small prediction tasks, prototyping, or cases where maintaining one trained model per task is not practical.

## Getting Started

To get started with a specific best practice:

1. Navigate to the relevant subdirectory
2. Read the README.md file for an overview and instructions
3. Explore the example implementations in your preferred programming language
4. Follow the setup instructions to run the examples in your own environment

## Contributing

If you'd like to contribute to these best practices:

1. Ensure your contribution follows the established patterns
2. Include documentation explaining the purpose and implementation details
3. Provide examples in at least one programming language
4. Submit a pull request with a clear description of your contribution
