"""
Progressive Drill-Down Analysis Module
Implements the Rivian Flux Analysis Requirements logic:
1. Detect GL fluctuation
2. Validate against consolidated reporting  
3. Decompose into drivers
4. Trace to lowest possible source (WBS → Project → Program → Vendor → PO)
5. Explain using structured + unstructured data
6. Escalate if evidence insufficient
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DrillDownResult:
    """Represents the result of a progressive drill-down analysis"""
    gl_account: str
    cost_center: str
    driver: str
    amount: float
    drill_path: List[str]  # The drill-down path taken
    evidence_chain: Dict[str, any]  # Evidence collected at each level
    root_cause: str  # Business explanation
    confidence: str  # High, Medium, Low
    requires_escalation: bool
    escalation_reason: Optional[str]
    missing_data: List[str]


class DrillDownAnalyzer:
    """
    Performs progressive drill-down analysis following the investigation hierarchy:
    GL → WBS/Project → Program → Cost Center → Vendor → PO → Text Analysis
    """
    
    def __init__(self, wbs_lookup: Dict[str, Dict]):
        self.wbs_lookup = wbs_lookup
        
    def determine_investigation_path(self, gl_account: str, df: pd.DataFrame) -> List[str]:
        """
        Determines the appropriate investigation path based on GL type.
        Different GLs require different drill logic (per requirements doc).
        """
        # R&D / Prototyping / Software → WBS-heavy
        rd_patterns = ['630', '631', '632', '633']  # Common R&D GL prefixes
        
        # Payroll → Org/Headcount heavy
        payroll_patterns = ['640', '641']
        
        # Rent / Facilities → Vendor/Location heavy  
        facilities_patterns = ['652', '653']
        
        gl_str = str(gl_account)
        
        if any(gl_str.startswith(p) for p in rd_patterns):
            return ['WBS/Project', 'Program', 'Cost_Center', 'Text', 'Vendor']
        elif any(gl_str.startswith(p) for p in payroll_patterns):
            return ['Cost_Center', 'Org', 'Headcount']
        elif any(gl_str.startswith(p) for p in facilities_patterns):
            return ['Vendor', 'Location', 'Cost_Center', 'Text']
        else:
            # Default path
            return ['WBS/Project', 'Cost_Center', 'Vendor', 'Text']
    
    def drill_into_wbs_project(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Step 1: Drill into WBS/Project dimension
        Returns aggregated data and enrichment from WBS master
        """
        evidence = {
            'level': 'WBS/Project',
            'found': False,
            'data': None
        }
        
        # Look for WBS-related columns
        wbs_columns = ['WBS Element', 'Internal Order', 'Project']
        found_col = None
        
        for col in wbs_columns:
            if col in df.columns:
                found_col = col
                break
        
        if not found_col:
            evidence['missing'] = 'No WBS/Project column found'
            return evidence
        
        # Aggregate by WBS/Project
        grouped = df.groupby(found_col).agg({
            'Delta': 'sum',
            'Abs_Delta': 'sum',
            'Document Number': 'count'
        }).reset_index()
        
        grouped = grouped.sort_values('Abs_Delta', ascending=False)
        
        # Enrich with WBS descriptions and program mappings
        grouped['Description'] = grouped[found_col].astype(str).map(
            self.wbs_lookup['wbs_descriptions']
        )
        grouped['Program'] = grouped[found_col].astype(str).apply(
            lambda x: self.wbs_lookup['wbs_programs'].get(x, {}).get('Project_Program', '')
        )
        grouped['Detailed_Program'] = grouped[found_col].astype(str).apply(
            lambda x: self.wbs_lookup['wbs_programs'].get(x, {}).get('Detailed_Program', '')
        )
        
        evidence['found'] = True
        evidence['data'] = None  # Don't include full DataFrame
        # Convert to native Python types
        top_records = grouped.head(5).to_dict('records')
        evidence['top_contributors'] = [
            {k: (float(v) if isinstance(v, (np.integer, np.floating)) else 
                 int(v) if isinstance(v, np.integer) else str(v))
             for k, v in record.items()}
            for record in top_records
        ]
        
        return evidence
    
    def drill_into_cost_center(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Step 2: Drill into Cost Center dimension
        """
        evidence = {
            'level': 'Cost_Center',
            'found': False,
            'data': None
        }
        
        if 'Cost Center' not in df.columns:
            evidence['missing'] = 'No Cost Center column found'
            return evidence
        
        grouped = df.groupby('Cost Center').agg({
            'Delta': 'sum',
            'Abs_Delta': 'sum',
            'Document Number': 'count'
        }).reset_index()
        
        grouped = grouped.sort_values('Abs_Delta', ascending=False)
        
        evidence['found'] = True
        evidence['data'] = None  # Don't include full DataFrame
        # Convert to native Python types
        top_records = grouped.head(5).to_dict('records')
        evidence['top_contributors'] = [
            {k: (float(v) if isinstance(v, (np.floating,)) else 
                 int(v) if isinstance(v, np.integer) else str(v))
             for k, v in record.items()}
            for record in top_records
        ]
        
        return evidence
    
    def drill_into_vendor(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Step 3: Drill into Vendor dimension (if available)
        """
        evidence = {
            'level': 'Vendor',
            'found': False,
            'data': None
        }
        
        # Look for vendor-related columns
        vendor_cols = ['Vendor', 'Vendor Number', 'Partner', 'Supplier']
        found_col = None
        
        for col in vendor_cols:
            if col in df.columns:
                found_col = col
                break
        
        if not found_col:
            evidence['missing'] = 'No Vendor column found'
            return evidence
        
        grouped = df.groupby(found_col).agg({
            'Delta': 'sum',
            'Abs_Delta': 'sum',
            'Document Number': 'count'
        }).reset_index()
        
        grouped = grouped.sort_values('Abs_Delta', ascending=False)
        
        evidence['found'] = True
        evidence['data'] = None  # Don't include full DataFrame
        # Convert to native Python types
        top_records = grouped.head(5).to_dict('records')
        evidence['top_contributors'] = [
            {k: (float(v) if isinstance(v, (np.floating,)) else 
                 int(v) if isinstance(v, np.integer) else str(v))
             for k, v in record.items()}
            for record in top_records
        ]
        
        return evidence
    
    def analyze_text_evidence(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Step 4: Analyze unstructured text fields
        Looks at Text, PO descriptions, Journal Entry text, etc.
        """
        evidence = {
            'level': 'Text_Analysis',
            'found': False,
            'data': None
        }
        
        text_columns = ['Text', 'Description', 'PO_Text', 'JE_Text', 'Item_Text']
        available_cols = [col for col in text_columns if col in df.columns]
        
        if not available_cols:
            evidence['missing'] = 'No text columns found'
            return evidence
        
        # Analyze each text column
        text_insights = {}
        for col in available_cols:
            texts = df[df[col].notna()][col].astype(str)
            if len(texts) > 0:
                text_insights[col] = {
                    'unique_count': int(len(texts.unique())),
                    'total_count': int(len(texts)),
                    'top_5': {k: int(v) for k, v in texts.value_counts().head(5).to_dict().items()}
                }
        
        evidence['found'] = True
        evidence['data'] = text_insights
        
        return evidence
    
    def assess_evidence_sufficiency(self, evidence_chain: Dict[str, any], 
                                    drill_path: List[str]) -> Tuple[bool, List[str]]:
        """
        Determines if we have sufficient evidence to explain the fluctuation.
        Returns: (is_sufficient, missing_data_list)
        """
        missing_data = []
        evidence_score = 0
        
        # Check each level in the drill path
        for level in drill_path:
            level_key = level.replace('/', '_')
            if level_key in evidence_chain:
                if evidence_chain[level_key].get('found', False):
                    evidence_score += 1
                else:
                    missing_data.append(level)
        
        # Need at least 2 levels of evidence for sufficiency
        is_sufficient = evidence_score >= 2
        
        return is_sufficient, missing_data
    
    def generate_root_cause_explanation(self, evidence_chain: Dict[str, any],
                                       gl_account: str, amount: float,
                                       driver: str) -> str:
        """
        Generates a business-readable explanation based on evidence collected.
        """
        explanation_parts = []
        
        # Ensure amount is float
        try:
            amount_val = float(amount)
            explanation_parts.append(
                f"GL Account {gl_account} shows a {driver} variance of ${amount_val:,.2f}."
            )
        except (ValueError, TypeError):
            explanation_parts.append(
                f"GL Account {gl_account} shows a {driver} variance of ${amount}."
            )
        
        # WBS/Project level
        if 'WBS_Project' in evidence_chain and evidence_chain['WBS_Project']['found']:
            wbs_data = evidence_chain['WBS_Project']
            top = wbs_data['top_contributors'][0] if wbs_data['top_contributors'] else None
            if top:
                try:
                    abs_delta = float(top.get('Abs_Delta', 0))
                    explanation_parts.append(
                        f"Primary driver: Project/WBS with {top.get('Description', 'N/A')} "
                        f"contributing ${abs_delta:,.2f}."
                    )
                except (ValueError, TypeError):
                    explanation_parts.append(
                        f"Primary driver: Project/WBS with {top.get('Description', 'N/A')}."
                    )
                if top.get('Program'):
                    explanation_parts.append(
                        f"This is part of program: {top['Program']}."
                    )
        
        # Cost Center level
        if 'Cost_Center' in evidence_chain and evidence_chain['Cost_Center']['found']:
            cc_data = evidence_chain['Cost_Center']
            top = cc_data['top_contributors'][0] if cc_data['top_contributors'] else None
            if top:
                try:
                    abs_delta = float(top.get('Abs_Delta', 0))
                    explanation_parts.append(
                        f"Primary cost center: {top.get('Cost Center', 'N/A')} "
                        f"with ${abs_delta:,.2f} variance."
                    )
                except (ValueError, TypeError):
                    explanation_parts.append(
                        f"Primary cost center: {top.get('Cost Center', 'N/A')}."
                    )
        
        # Vendor level
        if 'Vendor' in evidence_chain and evidence_chain['Vendor']['found']:
            vendor_data = evidence_chain['Vendor']
            top = vendor_data['top_contributors'][0] if vendor_data['top_contributors'] else None
            if top:
                vendor_cols = [k for k in top.keys() if 'Vendor' in k or 'Partner' in k]
                if vendor_cols:
                    vendor_col = vendor_cols[0]
                    try:
                        abs_delta = float(top.get('Abs_Delta', 0))
                        explanation_parts.append(
                            f"Key vendor/partner: {top[vendor_col]} "
                            f"contributing ${abs_delta:,.2f}."
                        )
                    except (ValueError, TypeError):
                        explanation_parts.append(
                            f"Key vendor/partner: {top[vendor_col]}."
                        )
        
        # Text analysis
        if 'Text_Analysis' in evidence_chain and evidence_chain['Text_Analysis']['found']:
            text_data = evidence_chain['Text_Analysis']['data']
            if text_data:
                for col, insights in text_data.items():
                    if insights['top_5']:
                        top_text = list(insights['top_5'].keys())[0]
                        count = insights['top_5'][top_text]
                        explanation_parts.append(
                            f"Common description: '{top_text}' appears {count} times."
                        )
                        break
        
        return " ".join(explanation_parts)
    
    def perform_drilldown(self, gl_account: str, cost_center: str, driver: str,
                         amount: float, df: pd.DataFrame) -> DrillDownResult:
        """
        Performs complete progressive drill-down analysis for a single GL/CC/Driver combination.
        """
        # Determine investigation path
        drill_path = self.determine_investigation_path(gl_account, df)
        
        # Execute drill-down following the path
        evidence_chain = {}
        
        if 'WBS/Project' in drill_path:
            evidence_chain['WBS_Project'] = self.drill_into_wbs_project(df)
        
        if 'Cost_Center' in drill_path:
            evidence_chain['Cost_Center'] = self.drill_into_cost_center(df)
        
        if 'Vendor' in drill_path:
            evidence_chain['Vendor'] = self.drill_into_vendor(df)
        
        if 'Text' in drill_path:
            evidence_chain['Text_Analysis'] = self.analyze_text_evidence(df)
        
        # Assess sufficiency
        is_sufficient, missing_data = self.assess_evidence_sufficiency(
            evidence_chain, drill_path
        )
        
        # Generate explanation
        root_cause = self.generate_root_cause_explanation(
            evidence_chain, gl_account, amount, driver
        )
        
        # Determine confidence
        evidence_count = sum(1 for e in evidence_chain.values() if e.get('found', False))
        if evidence_count >= 3:
            confidence = "High"
        elif evidence_count >= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Escalation logic
        requires_escalation = not is_sufficient or confidence == "Low"
        escalation_reason = None
        if requires_escalation:
            if not is_sufficient:
                escalation_reason = f"Insufficient evidence. Missing: {', '.join(missing_data)}"
            else:
                escalation_reason = "Low confidence in explanation. Requires business input."
        
        return DrillDownResult(
            gl_account=str(gl_account),
            cost_center=str(cost_center),
            driver=str(driver),
            amount=float(amount),
            drill_path=drill_path,
            evidence_chain=evidence_chain,
            root_cause=root_cause,
            confidence=confidence,
            requires_escalation=requires_escalation,
            escalation_reason=escalation_reason,
            missing_data=missing_data
        )


def perform_comprehensive_drilldown(summary_df: pd.DataFrame, 
                                   classified_df: pd.DataFrame,
                                   wbs_lookup: Dict[str, Dict],
                                   top_n: int = 20) -> List[DrillDownResult]:
    """
    Performs drill-down analysis on the top N variance items.
    Returns a list of DrillDownResult objects.
    """
    analyzer = DrillDownAnalyzer(wbs_lookup)
    results = []
    
    for idx, row in summary_df.head(top_n).iterrows():
        gl_account = row['Account_Number']
        cost_center = row['Cost_Center']
        driver = row['Driver']
        amount = row['Total_Delta']
        
        # Filter classified_df to this specific combination
        mask = (
            (classified_df['Account Number'] == gl_account) &
            (classified_df['Cost Center'] == cost_center) &
            (classified_df['Driver'] == driver)
        )
        subset_df = classified_df[mask]
        
        if len(subset_df) > 0:
            result = analyzer.perform_drilldown(
                gl_account, cost_center, driver, amount, subset_df
            )
            results.append(result)
    
    return results
