#############################################################################
# GEN AI HUB SETUP
#############################################################################
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

load_dotenv()

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from src.load_wbs import create_wbs_lookup
from src.drilldown_analysis import perform_comprehensive_drilldown, DrillDownResult

proxy_client = get_proxy_client("gen-ai-hub")


#############################################################################
# VARIANCE ANALYSIS FUNCTIONS
#############################################################################

def summarize_variances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes variances by Account Number (GL Account) and Cost Center.
    Groups document-level data into aggregated insights.
    """
    summary = df.groupby(["Account Number", "Cost Center", "Driver"]).agg({
        "Delta": "sum",
        "Abs_Delta": "sum",
        "Document Number": "count"
    }).reset_index()
    
    summary.columns = ["Account_Number", "Cost_Center", "Driver", "Total_Delta", "Total_Abs_Delta", "Document_Count"]
    summary = summary.sort_values("Total_Abs_Delta", ascending=False)
    
    return summary


def classify_variances_by_magnitude(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Classifies variances into categories based on magnitude:
    - Critical: Top 10% by absolute value
    - High: 10-30%
    - Medium: 30-60%
    - Low: Bottom 40%
    """
    df = df.copy()
    df = df.sort_values("Abs_Delta", ascending=False).reset_index(drop=True)
    
    total_rows = len(df)
    
    classifications = {
        "Critical": df.iloc[:int(total_rows * 0.10)],
        "High": df.iloc[int(total_rows * 0.10):int(total_rows * 0.30)],
        "Medium": df.iloc[int(total_rows * 0.30):int(total_rows * 0.60)],
        "Low": df.iloc[int(total_rows * 0.60):]
    }
    
    return classifications


def detect_patterns(df: pd.DataFrame) -> Dict[str, any]:
    """
    Detects patterns in variance data:
    - Most common drivers
    - GL accounts with highest variance concentration
    - Cost centers with highest variance concentration
    - Temporal patterns (if dates available)
    - Sign patterns (increases vs decreases)
    """
    patterns = {}
    
    # Driver distribution
    driver_counts = {k: int(v) for k, v in df["Driver"].value_counts().to_dict().items()}
    driver_amounts = {k: float(v) for k, v in df.groupby("Driver")["Delta"].sum().to_dict().items()}
    patterns["driver_distribution"] = {
        "counts": driver_counts,
        "total_amounts": driver_amounts
    }
    
    # Top GL accounts by variance
    gl_variance = df.groupby("Account Number")["Abs_Delta"].sum().sort_values(ascending=False)
    patterns["top_gl_accounts"] = {k: float(v) for k, v in gl_variance.head(10).to_dict().items()}
    
    # Top Cost Centers by variance
    cc_variance = df.groupby("Cost Center")["Abs_Delta"].sum().sort_values(ascending=False)
    patterns["top_cost_centers"] = {k: float(v) for k, v in cc_variance.head(10).to_dict().items()}
    
    # Sign analysis
    positive_delta = float(df[df["Delta"] > 0]["Abs_Delta"].sum())
    negative_delta = float(df[df["Delta"] < 0]["Abs_Delta"].sum())
    patterns["sign_analysis"] = {
        "total_increases": positive_delta,
        "total_decreases": negative_delta,
        "net_variance": float(df["Delta"].sum())
    }
    
    # Concentration analysis
    total_variance = float(df["Abs_Delta"].sum())
    top_20_pct = float(df.nlargest(int(len(df) * 0.20), "Abs_Delta")["Abs_Delta"].sum())
    patterns["concentration"] = {
        "total_variance": total_variance,
        "top_20_percent_variance": top_20_pct,
        "concentration_ratio": float(top_20_pct / total_variance) if total_variance > 0 else 0.0
    }
    
    # Document count by driver
    patterns["documents_by_driver"] = {k: int(v) for k, v in df["Driver"].value_counts().to_dict().items()}
    
    return patterns


def analyze_text_descriptions(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyzes the Text field for patterns and common themes.
    Returns insights about transaction descriptions.
    """
    text_analysis = {}
    
    # Check if Text column exists
    if "Text" not in df.columns:
        return {"error": "No Text column found in data"}
    
    # Get non-null texts
    texts = df[df["Text"].notna()]["Text"].astype(str)
    
    if len(texts) == 0:
        return {"error": "No text descriptions found"}
    
    # Most common descriptions
    text_analysis["top_descriptions"] = {k: int(v) for k, v in texts.value_counts().head(20).to_dict().items()}
    text_analysis["unique_descriptions"] = int(len(texts.unique()))
    text_analysis["total_with_text"] = int(len(texts))
    
    # Group texts by driver
    text_by_driver = {}
    for driver in df["Driver"].unique():
        driver_texts = df[df["Driver"] == driver]["Text"].dropna().astype(str)
        if len(driver_texts) > 0:
            text_by_driver[driver] = {k: int(v) for k, v in driver_texts.value_counts().head(5).to_dict().items()}
    
    text_analysis["text_by_driver"] = text_by_driver
    
    return text_analysis


def trace_to_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traces variances to their lowest possible source by adding context fields.
    Enriches the dataframe with source information.
    """
    traced = df.copy()
    
    # Add source identification
    traced["Source_Level"] = "Document"
    
    # Identify if it's a reversal
    if "Reversed With" in traced.columns:
        traced.loc[traced["Reversed With"].notna(), "Source_Level"] = "Reversal"
    
    # Identify consolidation-level transactions
    if "Consolidation Unit" in traced.columns:
        traced["Is_Intercompany"] = traced["Consolidation Unit"] != traced.get("Partner Unit", traced["Consolidation Unit"])
    
    # Add posting period for temporal analysis
    if "Posting Period" in traced.columns:
        traced["Posting_Quarter"] = traced["Posting Period"].apply(
            lambda x: f"Q{int((x-1)//3 + 1)}" if pd.notna(x) else "Unknown"
        )
    
    return traced


def identify_insufficient_evidence(summary: pd.DataFrame, patterns: Dict[str, any], 
                                   text_analysis: Dict[str, any]) -> List[Dict]:
    """
    Identifies variances that require escalation due to insufficient evidence.
    Returns a list of items that need cost center owner clarification.
    """
    escalations = []
    
    # Criteria for escalation:
    # 1. Large variances (top 5%) without clear text description
    # 2. New postings or reversals above threshold
    # 3. High concentration in single cost center without explanation
    
    threshold_amount = patterns["concentration"]["total_variance"] * 0.05
    
    for _, row in summary.iterrows():
        reasons = []
        
        # Check if variance is large
        if row["Total_Abs_Delta"] >= threshold_amount:
            # Check if driver suggests unclear origin
            if row["Driver"] in ["New Posting", "Reversal", "Accrual / Correction"]:
                reasons.append(f"Large {row['Driver']} without clear business justification")
            
            # Check text coverage
            if "error" not in text_analysis:
                driver_texts = text_analysis.get("text_by_driver", {}).get(row["Driver"], {})
                if len(driver_texts) == 0:
                    reasons.append("No description available for variance")
        
        if reasons:
            escalations.append({
                "Account_Number": str(row["Account_Number"]),
                "Cost_Center": str(row["Cost_Center"]),
                "Driver": str(row["Driver"]),
                "Amount": float(row["Total_Delta"]),
                "Document_Count": int(row["Document_Count"]),
                "Reasons": reasons,
                "Priority": "High" if row["Total_Abs_Delta"] >= threshold_amount * 2 else "Medium"
            })
    
    return escalations


def generate_insights_summary(patterns: Dict[str, any], text_analysis: Dict[str, any]) -> str:
    """
    Generates a human-readable summary of detected patterns.
    """
    summary_lines = []
    summary_lines.append("=== VARIANCE ANALYSIS INSIGHTS ===\n")
    
    # Driver insights
    summary_lines.append("TOP VARIANCE DRIVERS:")
    driver_amounts = patterns["driver_distribution"]["total_amounts"]
    for driver, amount in sorted(driver_amounts.items(), key=lambda x: abs(x[1]), reverse=True):
        summary_lines.append(f"  - {driver}: ${amount:,.2f}")
    
    summary_lines.append("\nDOCUMENT COUNT BY DRIVER:")
    for driver, count in patterns["documents_by_driver"].items():
        summary_lines.append(f"  - {driver}: {count} documents")
    
    # GL Account concentration
    summary_lines.append("\nTOP GL ACCOUNTS (by variance):")
    for gl, variance in list(patterns["top_gl_accounts"].items())[:5]:
        summary_lines.append(f"  - GL {gl}: ${variance:,.2f}")
    
    # Cost Center concentration
    summary_lines.append("\nTOP COST CENTERS (by variance):")
    for cc, variance in list(patterns["top_cost_centers"].items())[:5]:
        summary_lines.append(f"  - CC {cc}: ${variance:,.2f}")
    
    # Sign analysis
    sign_data = patterns["sign_analysis"]
    summary_lines.append("\nVARIANCE DIRECTION:")
    summary_lines.append(f"  - Total Increases: ${sign_data['total_increases']:,.2f}")
    summary_lines.append(f"  - Total Decreases: ${sign_data['total_decreases']:,.2f}")
    summary_lines.append(f"  - Net Variance: ${sign_data['net_variance']:,.2f}")
    
    # Concentration
    conc = patterns["concentration"]
    summary_lines.append("\nVARIANCE CONCENTRATION:")
    summary_lines.append(f"  - Total Variance: ${conc['total_variance']:,.2f}")
    summary_lines.append(f"  - Top 20% of items account for: {conc['concentration_ratio']*100:.1f}% of total variance")
    
    # Text analysis
    if "error" not in text_analysis:
        summary_lines.append("\nTEXT DESCRIPTION ANALYSIS:")
        summary_lines.append(f"  - Unique descriptions: {text_analysis['unique_descriptions']:,}")
        summary_lines.append(f"  - Items with descriptions: {text_analysis['total_with_text']:,}")
        summary_lines.append("\n  Top 5 descriptions:")
        for desc, count in list(text_analysis["top_descriptions"].items())[:5]:
            summary_lines.append(f"    - {desc}: {count} occurrences")
    
    return "\n".join(summary_lines)


#############################################################################
# LLM EXPLANATION FUNCTIONS
#############################################################################

def build_variance_context(summary_df: pd.DataFrame, patterns: Dict[str, any], 
                          text_analysis: Dict[str, any]) -> str:
    """
    Builds a structured context string for the LLM based on variance data and patterns.
    """
    context = "VARIANCE SUMMARY BY GL ACCOUNT AND COST CENTER:\n"
    
    # Add top 20 summarized variances
    for idx, row in summary_df.head(20).iterrows():
        context += (
            f"- GL Account {row['Account_Number']} | Cost Center {row['Cost_Center']} | "
            f"Driver: {row['Driver']} | Total Delta: ${row['Total_Delta']:,.2f} | "
            f"Document Count: {row['Document_Count']}\n"
        )
    
    context += "\n" + generate_insights_summary(patterns, text_analysis)
    
    # Add text description insights
    if "error" not in text_analysis and "text_by_driver" in text_analysis:
        context += "\n\nCOMMON TRANSACTION DESCRIPTIONS BY DRIVER:\n"
        for driver, texts in text_analysis["text_by_driver"].items():
            context += f"\n{driver}:\n"
            for desc, count in list(texts.items())[:3]:
                context += f"  - '{desc}' ({count} times)\n"
    
    return context


def build_explanation_prompt(context: str) -> str:
    """
    Builds the prompt for the LLM to explain variances.
    """
    prompt = f"""You are a senior finance controller analyzing R&D cost variances between two reporting periods (YTD Q3 2024 vs YTD Q3 2025).

Your task is to provide a clear, concise business explanation of the variances based on the data provided below.

INSTRUCTIONS:
1. Identify the main drivers of variance
2. Explain the business implications of these variances
3. Highlight any concerning patterns or anomalies
4. Analyze the transaction descriptions to understand what's behind the numbers
5. Provide actionable insights for management
6. Be specific and reference the actual numbers and drivers
7. Use professional business language
8. Keep the explanation between 300-500 words

DATA AND PATTERNS:
{context}

Provide your analysis:"""
    
    return prompt


def ask_llm_for_explanation(prompt: str) -> str:
    """
    Calls the SAP Generative AI Hub to get an LLM-generated explanation.
    """
    try:
        llm = ChatOpenAI(proxy_model_name="gpt-5.4", proxy_client=proxy_client)
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        return f"Error calling LLM: {str(e)}\n\nPlease check your Gen AI Hub configuration."


def ask_llm_simple(prompt: str) -> str:
    """
    Simple LLM invocation for general queries.
    """
    try:
        llm = ChatOpenAI(proxy_model_name="gpt-5.4", proxy_client=proxy_client)
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


#############################################################################
# MAIN ANALYSIS PIPELINE
#############################################################################

def analyze_variances_with_llm(classified_df: pd.DataFrame) -> Dict[str, any]:
    """
    Complete variance analysis pipeline following the workflow:
    1. Detect GL fluctuation (already done via classify_drivers)
    2. Validate against consolidated reporting (data structure)
    3. Decompose into drivers (already done)
    4. Trace to lowest possible source with WBS enrichment
    5. Perform progressive drill-down (GL → WBS → Project → Program)
    6. Explain using structured + unstructured data (Text descriptions)
    7. Summarize + escalate if evidence is insufficient
    
    Returns a dictionary with all analysis results.
    """
    # Step 1-3: Already done by previous functions (detect_variance, classify_drivers)
    
    # Load WBS lookup data for enrichment
    try:
        wbs_lookup = create_wbs_lookup()
    except Exception as e:
        print(f"Warning: Could not load WBS data: {e}")
        wbs_lookup = {'wbs_descriptions': {}, 'wbs_programs': {}}
    
    # Step 4: Trace to source
    traced = trace_to_source(classified_df)
    
    # Step 5a: Summarize
    summary = summarize_variances(traced)
    
    # Step 5b: Classify by magnitude
    classifications = classify_variances_by_magnitude(traced)
    
    # Step 5c: Detect patterns
    patterns = detect_patterns(traced)
    
    # Step 5d: Analyze text descriptions
    text_analysis = analyze_text_descriptions(traced)
    
    # Step 5e: Perform progressive drill-down on top variances
    drilldown_results = []
    if wbs_lookup['wbs_descriptions']:  # Only if WBS data loaded successfully
        try:
            drilldown_results = perform_comprehensive_drilldown(
                summary, traced, wbs_lookup, top_n=10
            )
        except Exception as e:
            print(f"Warning: Drill-down analysis failed: {e}")
    
    # Step 5f: Build context and get LLM explanation
    context = build_variance_context(summary, patterns, text_analysis)
    
    # Enrich context with drill-down insights if available
    if drilldown_results:
        context += "\n\nPROGRESSIVE DRILL-DOWN ANALYSIS (Top Variances):\n"
        for i, result in enumerate(drilldown_results[:5], 1):
            context += f"\n{i}. {result.root_cause}"
            context += f"\n   Confidence: {result.confidence} | Path: {' → '.join(result.drill_path)}"
            if result.requires_escalation:
                context += f"\n   ⚠️ Escalation needed: {result.escalation_reason}"
    
    prompt = build_explanation_prompt(context)
    llm_explanation = ask_llm_for_explanation(prompt)
    
    # Step 5g: Generate insights summary
    insights_summary = generate_insights_summary(patterns, text_analysis)
    
    # Step 6: Identify items needing escalation
    escalations = identify_insufficient_evidence(summary, patterns, text_analysis)
    
    # Add drill-down escalations
    for result in drilldown_results:
        if result.requires_escalation:
            escalations.append({
                "Account_Number": result.gl_account,
                "Cost_Center": result.cost_center,
                "Driver": result.driver,
                "Amount": result.amount,
                "Document_Count": 0,  # Changed from "N/A" to 0 for Arrow compatibility
                "Reasons": [result.escalation_reason],
                "Priority": "High" if result.confidence == "Low" else "Medium",
                "Missing_Data": ", ".join(result.missing_data) if result.missing_data else ""
            })
    
    return {
        "summary": summary,
        "classifications": classifications,
        "patterns": patterns,
        "text_analysis": text_analysis,
        "insights_summary": insights_summary,
        "llm_explanation": llm_explanation,
        "escalations": escalations,
        "drilldown_results": drilldown_results,
        "traced_data": traced,
        "context_used": context
    }
