"""
Summary report generation module for pharmaceutical anomaly detection.

This module creates comprehensive summary reports and analysis outputs.
"""

import os
import pandas as pd
from datetime import datetime
from typing import List
import config.settings as settings


def generate_summary_report(results_df: pd.DataFrame, feature_columns: List[str], model_type: str, 
                           n_estimators: int = None, max_samples = None, contamination_rate = None) -> None:
    """
    Generate a comprehensive summary report.
    
    Args:
        results_df: Results with predictions
        feature_columns: Feature column names  
        model_type: Type of model used
        n_estimators: Number of estimators used (if available)
        max_samples: Max samples per tree used (if available)
        contamination_rate: Contamination rate used (if available)
    """
    print(f"\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    report_lines = []
    report_lines.append("PHARMACEUTICAL ANOMALY DETECTION SUMMARY")
    report_lines.append("="*50)
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Model Type: Isolation Forest ({model_type})")
    report_lines.append(f"Dataset: {settings.CSV_FILENAME}")
    report_lines.append("")
    
    # Model Configuration
    report_lines.append("MODEL CONFIGURATION:")
    
    # Use actual parameters if provided, otherwise fall back to settings
    actual_n_estimators = n_estimators if n_estimators is not None else settings.N_ESTIMATORS
    actual_max_samples = max_samples if max_samples is not None else settings.MAX_SAMPLES
    actual_contamination_rate = contamination_rate if contamination_rate is not None else settings.CONTAMINATION_RATE
    
    report_lines.append(f"  - Number of Trees: {actual_n_estimators}")
    report_lines.append(f"  - Max Samples per Tree: {actual_max_samples}")
    
    if actual_contamination_rate == 'auto':
        detected_contamination = results_df['predicted_anomaly'].mean()
        report_lines.append(f"  - Contamination Mode: Auto (detected {detected_contamination:.1%})")
    else:
        report_lines.append(f"  - Expected Contamination Rate: {actual_contamination_rate:.1%}")
    report_lines.append(f"  - Features Used: {len(feature_columns)}")
    report_lines.append("")
    
    # Results Summary
    total_samples = len(results_df)
    anomaly_count = results_df['predicted_anomaly'].sum()
    anomaly_rate = anomaly_count / total_samples
    
    report_lines.append("DETECTION RESULTS:")
    report_lines.append(f"  - Total Samples Analyzed: {total_samples:,}")
    report_lines.append(f"  - Anomalies Detected: {anomaly_count:,} ({anomaly_rate:.2%})")
    report_lines.append(f"  - Normal Samples: {total_samples - anomaly_count:,} ({1-anomaly_rate:.2%})")
    report_lines.append("")
    
    # High-Risk Samples (Note: Lower anomaly scores indicate higher risk in Isolation Forest)
    if anomaly_count > 0:
        high_risk_samples = results_df[results_df['predicted_anomaly'] == 1].nsmallest(5, 'anomaly_score')
        
        report_lines.append("TOP 5 HIGHEST RISK SAMPLES:")
        for idx, (_, row) in enumerate(high_risk_samples.iterrows(), 1):
            doc_num = row.get(settings.KEY_COL, 'Unknown')
            score = row.get('anomaly_score', 0)
            explanation = row.get('anomaly_explanation', 'No explanation available')
            
            # Handle NaN values in explanation
            if pd.isna(explanation) or explanation is None:
                explanation = 'No explanation available'
            else:
                explanation = str(explanation)
                
            report_lines.append(f"  {idx}. Document {doc_num}: Score={score:.3f}")
            report_lines.append(f"     {explanation}...")
        report_lines.append("")
    
    # Feature Insights
    if 'anomaly_explanation' in results_df.columns:
        # Filter out NaN values before counting
        anomaly_explanations = results_df[results_df['predicted_anomaly'] == 1]['anomaly_explanation']
        explanations = anomaly_explanations.dropna().value_counts()
        if len(explanations) > 0:
            report_lines.append("MOST COMMON ANOMALY PATTERNS:")
            for explanation, count in explanations.head(5).items():
                if explanation and str(explanation) != 'nan':
                    explanation_str = str(explanation)[:80]
                    report_lines.append(f"  - {count:3d} cases: {explanation_str}...")
            report_lines.append("")
    
    # SHAP Insights (if available)
    if hasattr(results_df, 'attrs') and 'shap_explanations' in results_df.attrs:
        shap_data = results_df.attrs['shap_explanations']
        
        report_lines.append("SHAP-BASED FEATURE INSIGHTS:")
        
        # Global feature importance from SHAP
        if 'feature_importance' in shap_data:
            report_lines.append("  Most Important Features (Global SHAP Analysis):")
            for i, feat_info in enumerate(shap_data['feature_importance'][:5], 1):
                feature_name = feat_info['feature'].replace('is_', '').replace('_', ' ').title()
                importance = feat_info['importance']
                report_lines.append(f"    {i}. {feature_name}: {importance:.3f}")
            report_lines.append("")
        
        # Individual anomaly explanations
        if 'explanations' in shap_data and len(shap_data['explanations']) > 0:
            report_lines.append("  SHAP Explanations for Top Anomalies:")
            for i, exp in enumerate(shap_data['explanations'][:3], 1):
                doc_num = exp['document_number']
                score = exp['anomaly_score']
                shap_exp = exp['shap_explanation'][:100] + "..." if len(exp['shap_explanation']) > 100 else exp['shap_explanation']
                report_lines.append(f"    {i}. Document {doc_num} (Score: {score:.3f}):")
                report_lines.append(f"       {shap_exp}")
            report_lines.append("")
    
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("  1. Review high-scoring samples for potential issues")
    report_lines.append("  2. Investigate recurring anomaly patterns")
    report_lines.append("  3. Consider updating business rules based on findings")
    report_lines.append("  4. Monitor anomaly rates over time for trends")
    if hasattr(results_df, 'attrs') and 'shap_explanations' in results_df.attrs:
        report_lines.append("  5. Use SHAP feature importance to focus on key anomaly drivers")
        report_lines.append("  6. Review SHAP visualizations for detailed feature analysis")
    report_lines.append("")
    
    # Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(os.path.join(settings.RESULTS_DIR, 'summary_report.txt'), 'w') as f:
        f.write(report_text)
    
    print(f"Summary report saved to: {os.path.join(settings.RESULTS_DIR, 'summary_report.txt')}")