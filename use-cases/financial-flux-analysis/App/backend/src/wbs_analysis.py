"""
WBS Analysis Module
Performs variance analysis specifically for accounting entries with WBS elements.
The analysis is organized by WBS element as the main grouping factor.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.load_wbs import create_wbs_lookup, enrich_with_wbs_data
from src.explain_llm import ask_llm_simple


def filter_wbs_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataframe to include only transactions that have WBS elements.
    Since ACDOCU doesn't have explicit WBS columns, we use Cost Center as a proxy.
    We treat Cost Center as the WBS element for analysis purposes.
    """
    # Use Cost Center as WBS proxy - filter out empty cost centers
    mask = (
        df['Cost Center'].notna() & 
        (df['Cost Center'].astype(str) != '') & 
        (df['Cost Center'].astype(str) != '0') &
        (df['Cost Center'].astype(str) != 'nan')
    )
    
    return df[mask].copy()


def align_wbs_periods(df24: pd.DataFrame, df25: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns WBS transactions from both periods.
    Groups by Cost Center (as WBS proxy) to create comparable records.
    """
    # Filter only WBS transactions (Cost Centers with values)
    wbs24 = filter_wbs_transactions(df24)
    wbs25 = filter_wbs_transactions(df25)
    
    if wbs24.empty and wbs25.empty:
        return pd.DataFrame()
    
    # Use Cost Center as WBS element
    wbs_col = 'Cost Center'
    
    # Group by Cost Center (WBS proxy) and Account Number
    group_cols = [wbs_col, 'Account Number']
    
    # Aggregate by Cost Center (WBS) and GL Account
    agg_24 = wbs24.groupby(group_cols, dropna=False).agg({
        'Amount in Group Crcy': 'sum',
        'Document Number': 'count'
    }).reset_index()
    
    agg_25 = wbs25.groupby(group_cols, dropna=False).agg({
        'Amount in Group Crcy': 'sum',
        'Document Number': 'count'
    }).reset_index()
    
    # Rename columns
    agg_24.columns = ['WBS_Element', 'Account_Number', 'Amount_Q3_24', 'Doc_Count_24']
    agg_25.columns = ['WBS_Element', 'Account_Number', 'Amount_Q3_25', 'Doc_Count_25']
    
    # Merge on WBS (Cost Center) and GL Account
    aligned = pd.merge(
        agg_24, 
        agg_25, 
        on=['WBS_Element', 'Account_Number'],
        how='outer'
    )
    
    # Fill NaN values with 0
    aligned['Amount_Q3_24'] = aligned['Amount_Q3_24'].fillna(0)
    aligned['Amount_Q3_25'] = aligned['Amount_Q3_25'].fillna(0)
    aligned['Doc_Count_24'] = aligned['Doc_Count_24'].fillna(0)
    aligned['Doc_Count_25'] = aligned['Doc_Count_25'].fillna(0)
    
    # Calculate variance
    aligned['Delta'] = aligned['Amount_Q3_25'] - aligned['Amount_Q3_24']
    aligned['Abs_Delta'] = aligned['Delta'].abs()
    aligned['Percent_Change'] = np.where(
        aligned['Amount_Q3_24'] != 0,
        (aligned['Delta'] / aligned['Amount_Q3_24'].abs()) * 100,
        np.inf
    )
    
    return aligned


def enrich_wbs_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches WBS analysis with descriptions and program hierarchy.
    """
    if df.empty:
        return df
    
    try:
        wbs_lookup = create_wbs_lookup()
        
        # Add WBS descriptions
        df['WBS_Description'] = df['WBS_Element'].astype(str).map(
            wbs_lookup['wbs_descriptions']
        )
        
        # Add program hierarchy
        df['Project_Program'] = df['WBS_Element'].astype(str).apply(
            lambda x: wbs_lookup['wbs_programs'].get(x, {}).get('Project_Program', '')
        )
        df['Detailed_Program'] = df['WBS_Element'].astype(str).apply(
            lambda x: wbs_lookup['wbs_programs'].get(x, {}).get('Detailed_Program', '')
        )
        df['High_Level_Program'] = df['WBS_Element'].astype(str).apply(
            lambda x: wbs_lookup['wbs_programs'].get(x, {}).get('High_Level_Program', '')
        )
        
    except Exception as e:
        print(f"Warning: Could not enrich with WBS data: {e}")
        df['WBS_Description'] = ''
        df['Project_Program'] = ''
        df['Detailed_Program'] = ''
        df['High_Level_Program'] = ''
    
    return df


def classify_wbs_variances(df: pd.DataFrame, materiality_threshold: float = 1000) -> pd.DataFrame:
    """
    Classifies WBS variances and applies materiality threshold.
    """
    if df.empty:
        return df
    
    # Apply materiality filter
    df = df[df['Abs_Delta'] >= materiality_threshold].copy()
    
    # Classify variance direction
    df['Variance_Type'] = df['Delta'].apply(
        lambda x: 'Increase' if x > 0 else 'Decrease' if x < 0 else 'No Change'
    )
    
    # Classify by magnitude
    df = df.sort_values('Abs_Delta', ascending=False).reset_index(drop=True)
    total_rows = len(df)
    
    if total_rows > 0:
        df['Magnitude_Class'] = 'Low'
        df.loc[df.index < int(total_rows * 0.10), 'Magnitude_Class'] = 'Critical'
        df.loc[(df.index >= int(total_rows * 0.10)) & 
               (df.index < int(total_rows * 0.30)), 'Magnitude_Class'] = 'High'
        df.loc[(df.index >= int(total_rows * 0.30)) & 
               (df.index < int(total_rows * 0.60)), 'Magnitude_Class'] = 'Medium'
    
    return df


def summarize_by_wbs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary grouped by WBS Element (Cost Center) showing total variances.
    """
    if df.empty:
        return df
    
    # Group by WBS Element (Cost Center)
    group_cols = ['WBS_Element']
    
    # Add description and program columns if they exist
    if 'WBS_Description' in df.columns:
        group_cols.append('WBS_Description')
    if 'Project_Program' in df.columns:
        group_cols.append('Project_Program')
    if 'Detailed_Program' in df.columns:
        group_cols.append('Detailed_Program')
    if 'High_Level_Program' in df.columns:
        group_cols.append('High_Level_Program')
    
    summary = df.groupby(group_cols, dropna=False).agg({
        'Amount_Q3_24': 'sum',
        'Amount_Q3_25': 'sum',
        'Delta': 'sum',
        'Abs_Delta': 'sum',
        'Account_Number': 'count'
    }).reset_index()
    
    # Rename columns dynamically based on what exists
    col_mapping = {
        'Amount_Q3_24': 'Total_Q3_24',
        'Amount_Q3_25': 'Total_Q3_25',
        'Delta': 'Net_Variance',
        'Abs_Delta': 'Total_Abs_Variance',
        'Account_Number': 'GL_Account_Count'
    }
    summary = summary.rename(columns=col_mapping)
    
    summary = summary.sort_values('Total_Abs_Variance', ascending=False)
    
    return summary


def summarize_by_program(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary grouped by Program hierarchy.
    """
    if df.empty:
        return df
    
    # Filter out rows without program information
    df_with_program = df[df['High_Level_Program'].notna() & 
                         (df['High_Level_Program'] != '')].copy()
    
    if df_with_program.empty:
        return pd.DataFrame()
    
    summary = df_with_program.groupby(['High_Level_Program', 'Detailed_Program', 
                                       'Project_Program']).agg({
        'Amount_Q3_24': 'sum',
        'Amount_Q3_25': 'sum',
        'Delta': 'sum',
        'Abs_Delta': 'sum',
        'WBS_Element': 'count'
    }).reset_index()
    
    summary.columns = ['High_Level_Program', 'Detailed_Program', 'Project_Program',
                       'Total_Q3_24', 'Total_Q3_25', 'Net_Variance', 
                       'Total_Abs_Variance', 'WBS_Count']
    
    summary = summary.sort_values('Total_Abs_Variance', ascending=False)
    
    return summary


def generate_wbs_insights(df: pd.DataFrame, wbs_summary: pd.DataFrame, 
                         program_summary: pd.DataFrame) -> str:
    """
    Generates a text summary of WBS variance insights.
    """
    insights = []
    insights.append("=== WBS VARIANCE ANALYSIS INSIGHTS ===\n")
    
    if df.empty:
        insights.append("No WBS transactions found in the dataset.")
        return "\n".join(insights)
    
    # Overall statistics
    total_variance = df['Delta'].sum()
    total_abs_variance = df['Abs_Delta'].sum()
    unique_wbs = df['WBS_Element'].nunique()
    
    insights.append(f"OVERALL STATISTICS:")
    insights.append(f"  - Unique WBS Elements: {unique_wbs:,}")
    insights.append(f"  - Total Net Variance: ${total_variance:,.2f}")
    insights.append(f"  - Total Absolute Variance: ${total_abs_variance:,.2f}")
    insights.append(f"  - Number of GL Account/WBS combinations: {len(df):,}")
    
    # Variance direction
    increases = df[df['Delta'] > 0]['Abs_Delta'].sum()
    decreases = df[df['Delta'] < 0]['Abs_Delta'].sum()
    insights.append(f"\nVARIANCE DIRECTION:")
    insights.append(f"  - Total Increases: ${increases:,.2f}")
    insights.append(f"  - Total Decreases: ${decreases:,.2f}")
    
    # Top WBS elements
    if not wbs_summary.empty:
        insights.append(f"\nTOP 10 WBS ELEMENTS BY VARIANCE:")
        for idx, row in wbs_summary.head(10).iterrows():
            desc = row['WBS_Description'] if row['WBS_Description'] else 'No Description'
            insights.append(
                f"  - {row['WBS_Element']}: ${row['Net_Variance']:,.2f} "
                f"(Q3'24: ${row['Total_Q3_24']:,.2f} → Q3'25: ${row['Total_Q3_25']:,.2f})"
            )
            insights.append(f"    Description: {desc}")
    
    # Top programs
    if not program_summary.empty:
        insights.append(f"\nTOP 10 PROGRAMS BY VARIANCE:")
        for idx, row in program_summary.head(10).iterrows():
            insights.append(
                f"  - {row['High_Level_Program']} / {row['Detailed_Program']}: "
                f"${row['Net_Variance']:,.2f} ({row['WBS_Count']} WBS elements)"
            )
    
    # Magnitude distribution
    if 'Magnitude_Class' in df.columns:
        insights.append(f"\nVARIANCE MAGNITUDE DISTRIBUTION:")
        for mag_class in ['Critical', 'High', 'Medium', 'Low']:
            count = len(df[df['Magnitude_Class'] == mag_class])
            if count > 0:
                insights.append(f"  - {mag_class}: {count} items")
    
    return "\n".join(insights)


def explain_wbs_variances_with_llm(df: pd.DataFrame, wbs_summary: pd.DataFrame,
                                   program_summary: pd.DataFrame, insights: str) -> str:
    """
    Uses LLM to generate an explanation of WBS variances.
    """
    if df.empty:
        return "No WBS transactions found for analysis."
    
    # Build context for LLM
    context = f"""WBS VARIANCE ANALYSIS DATA:

{insights}

DETAILED WBS BREAKDOWN (Top 15):
"""
    
    for idx, row in wbs_summary.head(15).iterrows():
        context += (
            f"\nWBS Element: {row['WBS_Element']}\n"
            f"  Description: {row['WBS_Description'] if row['WBS_Description'] else 'N/A'}\n"
            f"  Program: {row['High_Level_Program']} > {row['Detailed_Program']}\n"
            f"  Q3 2024: ${row['Total_Q3_24']:,.2f}\n"
            f"  Q3 2025: ${row['Total_Q3_25']:,.2f}\n"
            f"  Variance: ${row['Net_Variance']:,.2f}\n"
            f"  GL Accounts affected: {row['GL_Account_Count']}\n"
        )
    
    if not program_summary.empty:
        context += "\n\nPROGRAM LEVEL SUMMARY (Top 10):\n"
        for idx, row in program_summary.head(10).iterrows():
            context += (
                f"\nProgram: {row['High_Level_Program']} / {row['Detailed_Program']}\n"
                f"  Q3 2024: ${row['Total_Q3_24']:,.2f}\n"
                f"  Q3 2025: ${row['Total_Q3_25']:,.2f}\n"
                f"  Variance: ${row['Net_Variance']:,.2f}\n"
                f"  WBS Elements: {row['WBS_Count']}\n"
            )
    
    prompt = f"""You are a senior finance controller analyzing R&D cost variances for WBS (Work Breakdown Structure) elements between YTD Q3 2024 and YTD Q3 2025.

Your task is to provide a clear, executive-level explanation of the WBS variances based on the data provided below.

INSTRUCTIONS:
1. Identify the main WBS elements and programs driving the variance
2. Explain what these WBS elements represent and why their costs changed
3. Analyze the program-level view to identify strategic patterns
4. Highlight any concerning trends or anomalies at the project/program level
5. Provide actionable insights for R&D management
6. Reference specific WBS elements, programs, and amounts
7. Use professional business language suitable for executive presentation
8. Keep the explanation between 400-600 words
9. Structure your response with clear sections (Overview, Key Drivers, Program Analysis, Recommendations)

DATA:
{context}

Provide your analysis:"""
    
    try:
        explanation = ask_llm_simple(prompt)
        return explanation
    except Exception as e:
        return f"Error generating LLM explanation: {str(e)}\n\nPlease review the data manually above."


def perform_wbs_analysis(df24: pd.DataFrame, df25: pd.DataFrame, 
                        materiality_threshold: float = 1000,
                        run_llm: bool = True) -> Dict:
    """
    Main function to perform complete WBS variance analysis.
    
    Returns a dictionary with:
    - aligned_data: WBS-level variance data
    - wbs_summary: Summary by WBS element
    - program_summary: Summary by program hierarchy
    - insights: Text summary of insights
    - llm_explanation: AI-generated explanation (if run_llm=True)
    - critical_items: Items classified as critical
    """
    # Step 1: Align WBS transactions from both periods
    aligned = align_wbs_periods(df24, df25)
    
    if aligned.empty:
        return {
            "aligned_data": pd.DataFrame(),
            "wbs_summary": pd.DataFrame(),
            "program_summary": pd.DataFrame(),
            "insights": "No WBS transactions found in the dataset.",
            "llm_explanation": "No WBS transactions available for analysis.",
            "critical_items": pd.DataFrame(),
            "error": "No WBS transactions found"
        }
    
    # Step 2: Enrich with WBS descriptions and program hierarchy
    aligned = enrich_wbs_analysis(aligned)
    
    # Step 3: Classify variances
    aligned = classify_wbs_variances(aligned, materiality_threshold)
    
    # Step 4: Create summaries
    wbs_summary = summarize_by_wbs(aligned)
    program_summary = summarize_by_program(aligned)
    
    # Step 5: Generate insights
    insights = generate_wbs_insights(aligned, wbs_summary, program_summary)
    
    # Step 6: Get critical items
    critical_items = aligned[aligned['Magnitude_Class'] == 'Critical'].copy() if 'Magnitude_Class' in aligned.columns else pd.DataFrame()
    
    # Step 7: Generate LLM explanation if requested
    llm_explanation = ""
    if run_llm:
        llm_explanation = explain_wbs_variances_with_llm(
            aligned, wbs_summary, program_summary, insights
        )
    
    return {
        "aligned_data": aligned,
        "wbs_summary": wbs_summary,
        "program_summary": program_summary,
        "insights": insights,
        "llm_explanation": llm_explanation,
        "critical_items": critical_items
    }
