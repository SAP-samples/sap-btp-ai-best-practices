# Pharma Synthetic Data Prototype

This repository contains a **self-contained example** designed to generate and validate synthetic datasets for testing scenarios related to **upcoming U.S. pharmaceutical legislation changes** expected next year.  
The purpose of this project is to provide a reproducible prototype that allows teams to experiment with realistic data structures and validation rules without relying on sensitive or proprietary information.

---

## 📌 Key Features

1. **Synthetic Data Generation**  
   - Creates structured pharmaceutical sales and distribution records.  
   - Mimics real-world fields such as wholesaler, sales document type, order details, material identifiers, NDC codes, and pricing.  
   - Includes variability for testing edge cases (aberrant blocks, abnormal volumes, compliance flags, etc.).

2. **Validation Approaches**  
   - Built-in checks on data consistency (e.g., numeric ranges, averages, exceeded thresholds).  
   - Flags for conditions relevant to compliance, such as exceeding 90-day averages, customer block reasons, and order validation logic.  
   - Clear separation between **valid data**, **outliers**, and **legislative triggers**.

3. **Testing Focus**  
   - Supports test cases for evaluating processes impacted by pharmaceutical regulation updates.  
   - Demonstrates how machine learning models or rule-based systems can be applied to detect anomalies.  
   - Provides a foundation for discussions on automation and compliance monitoring.

---

## 🧩 Notebook Structure

The notebook is organized into three main steps:

1. **Run Synthetic Data Generation**  
   - Produces a dataset of pharmaceutical transactions with realistic patterns and anomalies.  

2. **Synthetic Data Validation**  
   - Applies validation rules to detect outliers, blocked customers, abnormal orders, and compliance thresholds (e.g., 90-day averages).  

3. **Enhanced ML Modelling**  
   - Demonstrates how machine learning methods can be applied on top of validation outputs to classify cases and support compliance automation.  

---

## 🔍 Relevance to U.S. Regulatory Changes

Beginning in **2026**, U.S. pharmaceutical distributors and manufacturers will be subject to **stricter requirements** under the **Drug Supply Chain Security Act (DSCSA)** full interoperability mandate and related FDA guidance.  
Key aspects include:  

- End-to-end electronic tracing of prescription drugs across the supply chain.  
- Enhanced monitoring of order volumes and outlier detection (e.g., excessive 90-day averages).  
- Tighter validation of wholesaler orders and customer blocks.  
- Clear auditability of exceptions, with penalties for non-compliance.  

These synthetic datasets allow stakeholders to **simulate how their systems will behave under the new rules**, test ML or rules-based anomaly detection, and explore workflows for handling compliance triggers without exposing real data.

---

## 📜 References to Normative Acts

- **Drug Supply Chain Security Act (DSCSA)** – [FDA DSCSA Guidance](https://www.fda.gov/drugs/drug-supply-chain-integrity/drug-supply-chain-security-act-dscsa)  
- **FDA Enhanced Drug Distribution Security Requirements (2023–2026)** – [FDA Guidance PDF](https://www.fda.gov/media/163761/download)  
- **Centers for Medicare & Medicaid Services (CMS) – Drug Price Transparency Rules** – [CMS Drug Transparency](https://www.cms.gov/medicare/drug-price-transparency)  

---

## 🚀 How to Run

1. **Environment Setup**
   - Install [Jupyter Notebook](https://jupyter.org/) or run in any Python environment supporting notebooks.  
   - Required Python packages:  
     ```bash
     pip install pandas numpy matplotlib
     ```
     (Additional dependencies, if any, are listed inside the notebook.)

2. **Open the Notebook**
   - Launch Jupyter and open the file:  
     ```
     Pharma Synthetic data 202050822_Handover_prototype.ipynb
     ```

3. **Execute Cells**
   - Run the notebook step by step to:  
     - Generate synthetic datasets.  
     - Apply validation rules.  
     - Run enhanced ML modelling.  

4. **Outputs**
   - Clean datasets ready for analysis or model testing.  
   - Highlighted validation results with flagged cases.  
   - Optional plots/tables to illustrate data distributions and anomalies.

---

## ✅ Summary

This notebook provides a **realistic, self-contained playground** for exploring upcoming U.S. pharmaceutical compliance challenges. It generates structured synthetic data, validates it through rule-based checks, and enables users to experiment with automation or model-driven detection approaches—all without depending on real-world data sources.
