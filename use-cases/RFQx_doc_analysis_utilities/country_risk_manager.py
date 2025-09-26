"""
Country Risk Data Management for RFQ Analysis Application.

This module handles loading, parsing, and querying country risk data from CSV files
to enhance RFQ analysis with country-specific risk insights.
"""

import csv
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from difflib import get_close_matches


class CountryRiskManager:
    """Manages country risk data loading, caching, and lookup functionality."""
    
    def __init__(self, csv_path: str = "extra_docs/risk_country.csv"):
        """
        Initialize the Country Risk Manager.
        
        Args:
            csv_path: Path to the country risk CSV file
        """
        self.csv_path = Path(csv_path)
        self.risk_data = {}
        self.country_mapping = {}
        self.risk_columns = []
        self._load_risk_data()
    
    def _load_risk_data(self) -> None:
        """Load and parse the country risk CSV data."""
        if not self.csv_path.exists():
            print(f"Warning: Country risk CSV not found at {self.csv_path}")
            return
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                # Handle BOM if present
                content = file.read()
                if content.startswith('\ufeff'):
                    content = content[1:]
                
                # Parse CSV
                reader = csv.DictReader(content.splitlines())
                
                # Define the risk columns we're interested in (from "Lack of Reliability" to "Access to health care")
                all_columns = reader.fieldnames
                start_idx = None
                end_idx = None
                
                for i, col in enumerate(all_columns):
                    if 'Lack of Reliability' in col:
                        start_idx = i
                    if 'Access to health care' in col:
                        end_idx = i + 1
                        break
                
                if start_idx is not None and end_idx is not None:
                    self.risk_columns = all_columns[start_idx:end_idx]
                else:
                    # Fallback to manually defined columns
                    self.risk_columns = [
                        'Lack of Reliability (*)', 'HAZARD & EXPOSURE', 'Natural', 'Earthquake',
                        'River Flood', 'Tsunami', 'Tropical Cyclone', 'Coastal flood',
                        'Drought', 'Epidemic', 'Human', 'Projected Conflict Probability',
                        'Current Conflict Intensity', 'VULNERABILITY',
                        'Socio-Economic Vulnerability', 'Development & Deprivation',
                        'Inequality', 'Economic Dependency', 'Vulnerable Groups',
                        'Uprooted people', 'Health Conditions', 'Children U5', 'Recent Shocks',
                        'Food Security', 'Other Vulnerable Groups', 'LACK OF COPING CAPACITY',
                        'Institutional', 'DRR', 'Governance', 'Infrastructure', 'Communication',
                        'Physical infrastructure', 'Access to health care'
                    ]
                
                # Load data for each country
                for row in reader:
                    country = row.get('COUNTRY', '').strip()
                    if country:
                        # Store main risk data
                        risk_info = {
                            'country': country,
                            'iso3': row.get('ISO3', ''),
                            'inform_risk': self._safe_float(row.get('INFORM RISK', '')),
                            'risk_class': row.get('RISK CLASS', ''),
                            'rank': self._safe_int(row.get('Rank', ''))
                        }
                        
                        # Add risk metrics (0-10 scores)
                        for col in self.risk_columns:
                            if col in row:
                                risk_info[self._normalize_column_name(col)] = self._safe_float(row[col])
                        
                        self.risk_data[country.lower()] = risk_info
                        
                        # Create country mapping for common variations
                        self._add_country_mappings(country)
                
                print(f"Loaded risk data for {len(self.risk_data)} countries")
                
        except Exception as e:
            print(f"Error loading country risk data: {e}")
    
    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float, handling 'x' and other non-numeric values."""
        if not value or value.strip() in ['x', 'X', '', 'N/A']:
            return None
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: str) -> Optional[int]:
        """Safely convert string to int."""
        if not value or value.strip() in ['x', 'X', '', 'N/A']:
            return None
        try:
            return int(float(value.strip()))
        except (ValueError, TypeError):
            return None
    
    def _normalize_column_name(self, column: str) -> str:
        """Normalize column name to a clean field name."""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s]', '', column)
        cleaned = re.sub(r'\s+', '_', cleaned.strip().lower())
        return cleaned
    
    def _add_country_mappings(self, country: str) -> None:
        """Add country name variations for better matching."""
        country_lower = country.lower()
        
        # Common mappings
        mappings = {
            'united kingdom': ['england', 'scotland', 'wales', 'uk', 'britain', 'great britain'],
            'united states of america': ['usa', 'us', 'america', 'united states'],
            'russian federation': ['russia'],
            'china': ['peoples republic of china', 'prc'],
            'south korea': ['republic of korea'],
            'north korea': ['democratic peoples republic of korea'],
            'vietnam': ['viet nam'],
            'czech republic': ['czechia'],
            'bosnia and herzegovina': ['bosnia'],
            'serbia': ['republic of serbia'],
        }
        
        # Add mappings for this country
        for main_country, variants in mappings.items():
            if country_lower == main_country:
                for variant in variants:
                    self.country_mapping[variant] = country_lower
            elif country_lower in variants:
                self.country_mapping[country_lower] = main_country
    
    def find_country_risk(self, country_name: str) -> Optional[Dict[str, Any]]:
        """
        Find country risk data for a given country name.
        
        Args:
            country_name: Name of the country to lookup
            
        Returns:
            Dictionary containing risk data or None if not found
        """
        if not country_name:
            return None
        
        country_lower = country_name.strip().lower()
        
        # Direct match
        if country_lower in self.risk_data:
            return self.risk_data[country_lower].copy()
        
        # Check mappings
        if country_lower in self.country_mapping:
            mapped_country = self.country_mapping[country_lower]
            if mapped_country in self.risk_data:
                return self.risk_data[mapped_country].copy()
        
        # Fuzzy matching as last resort
        all_countries = list(self.risk_data.keys())
        matches = get_close_matches(country_lower, all_countries, n=1, cutoff=0.8)
        if matches:
            return self.risk_data[matches[0]].copy()
        
        return None
    
    def get_risk_summary(self, country_name: str) -> Optional[str]:
        """
        Get a formatted risk summary for a country.
        
        Args:
            country_name: Name of the country
            
        Returns:
            Formatted string summary of risk data
        """
        risk_data = self.find_country_risk(country_name)
        if not risk_data:
            return None
        
        summary_lines = [
            f"**Country:** {risk_data['country']}",
            f"**Risk Class:** {risk_data.get('risk_class', 'Unknown')}",
            f"**Overall Risk Score:** {risk_data.get('inform_risk', 'N/A')}/10",
        ]
        
        # Add key risk metrics
        key_metrics = [
            ('lack_of_reliability', 'Lack of Reliability'),
            ('projected_conflict_probability', 'Projected Conflict Probability'),
            ('governance', 'Governance'),
            ('infrastructure', 'Infrastructure'),
            ('current_conflict_intensity', 'Current Conflict Intensity'),
        ]
        
        for field, label in key_metrics:
            value = risk_data.get(field)
            if value is not None:
                summary_lines.append(f"**{label}:** {value}/10")
        
        return "\n".join(summary_lines)
    
    def get_detailed_risk_analysis(self, country_name: str) -> Optional[str]:
        """
        Get a detailed risk analysis for inclusion in reports.
        
        Args:
            country_name: Name of the country
            
        Returns:
            Detailed markdown-formatted risk analysis
        """
        risk_data = self.find_country_risk(country_name)
        if not risk_data:
            return None
        
        country = risk_data['country']
        risk_class = risk_data.get('risk_class', 'Unknown')
        overall_risk = risk_data.get('inform_risk')
        
        analysis = [
            f"## Country Risk Analysis: {country}",
            "",
            f"The contracting authority is located in **{country}**, which has an overall risk classification of **{risk_class}**",
        ]
        
        if overall_risk is not None:
            analysis.append(f"with an INFORM risk score of **{overall_risk}/10**.")
        else:
            analysis.append(".")
        
        analysis.extend(["", "### Key Risk Indicators"])
        
        # Governance and institutional risks
        governance = risk_data.get('governance')
        institutional = risk_data.get('institutional')
        if governance is not None or institutional is not None:
            analysis.append("")
            analysis.append("**Governance & Institutional Environment:**")
            if governance is not None:
                risk_level = self._interpret_risk_score(governance)
                analysis.append(f"- Governance quality: {governance}/10 ({risk_level})")
            if institutional is not None:
                risk_level = self._interpret_risk_score(institutional)
                analysis.append(f"- Institutional capacity: {institutional}/10 ({risk_level})")
        
        # Security and conflict risks
        conflict_prob = risk_data.get('projected_conflict_probability')
        conflict_intensity = risk_data.get('current_conflict_intensity')
        if conflict_prob is not None or conflict_intensity is not None:
            analysis.append("")
            analysis.append("**Security & Conflict Environment:**")
            if conflict_prob is not None:
                risk_level = self._interpret_risk_score(conflict_prob)
                analysis.append(f"- Projected conflict probability: {conflict_prob}/10 ({risk_level})")
            if conflict_intensity is not None:
                risk_level = self._interpret_risk_score(conflict_intensity)
                analysis.append(f"- Current conflict intensity: {conflict_intensity}/10 ({risk_level})")
        
        # Infrastructure and operational risks
        infrastructure = risk_data.get('infrastructure')
        communication = risk_data.get('communication')
        if infrastructure is not None or communication is not None:
            analysis.append("")
            analysis.append("**Infrastructure & Operational Environment:**")
            if infrastructure is not None:
                risk_level = self._interpret_risk_score(infrastructure)
                analysis.append(f"- Infrastructure quality: {infrastructure}/10 ({risk_level})")
            if communication is not None:
                risk_level = self._interpret_risk_score(communication)
                analysis.append(f"- Communication infrastructure: {communication}/10 ({risk_level})")
        
        # Reliability and economic factors
        reliability = risk_data.get('lack_of_reliability')
        if reliability is not None:
            analysis.append("")
            analysis.append("**Reliability & Economic Factors:**")
            risk_level = self._interpret_risk_score(reliability)
            analysis.append(f"- Lack of reliability indicator: {reliability}/10 ({risk_level})")
        
        # Risk implications
        analysis.extend(["", "### Risk Implications for Contracting"])
        analysis.append(self._generate_risk_implications(risk_data))
        
        return "\n".join(analysis)
    
    def _interpret_risk_score(self, score: float) -> str:
        """Interpret a risk score (0-10) as risk level."""
        if score <= 2:
            return "Very Low Risk"
        elif score <= 4:
            return "Low Risk"
        elif score <= 6:
            return "Medium Risk"
        elif score <= 8:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _generate_risk_implications(self, risk_data: Dict[str, Any]) -> str:
        """Generate risk implications text based on the risk scores."""
        implications = []
        
        # Overall risk class implications
        risk_class = risk_data.get('risk_class', '').lower()
        if 'very high' in risk_class:
            implications.append("⚠️ **High Risk Environment**: This location presents significant challenges for contract execution and may require enhanced risk mitigation measures.")
        elif 'high' in risk_class:
            implications.append("⚠️ **Elevated Risk**: Careful risk assessment and mitigation planning are recommended for this contracting environment.")
        elif 'medium' in risk_class:
            implications.append("⚠️ **Moderate Risk**: Standard risk management practices should be sufficient, with monitoring of key risk indicators.")
        elif 'low' in risk_class or 'very low' in risk_class:
            implications.append("✅ **Low Risk Environment**: This location presents a relatively stable environment for contract execution.")
        
        # Specific risk factor implications
        governance = risk_data.get('governance', 0)
        if governance > 6:
            implications.append("- Governance challenges may impact regulatory compliance and project approvals.")
        
        conflict_prob = risk_data.get('projected_conflict_probability', 0)
        if conflict_prob > 4:
            implications.append("- Security considerations should be factored into project planning and personnel safety.")
        
        infrastructure = risk_data.get('infrastructure', 0)
        if infrastructure > 6:
            implications.append("- Infrastructure limitations may affect project logistics and implementation timelines.")
        
        reliability = risk_data.get('lack_of_reliability', 0)
        if reliability > 6:
            implications.append("- Enhanced due diligence and contract monitoring may be necessary due to reliability concerns.")
        
        if not implications:
            implications.append("The risk profile suggests standard contracting practices should be sufficient.")
        
        return "\n".join(implications)
    
    def get_countries_list(self) -> List[str]:
        """Get list of all available countries in the risk database."""
        return [data['country'] for data in self.risk_data.values()]
    
    def get_risk_context_for_llm(self, country_name: str) -> Optional[str]:
        """
        Get country risk context formatted for LLM inclusion.
        
        Args:
            country_name: Name of the country
            
        Returns:
            Formatted context string for LLM prompts
        """
        risk_data = self.find_country_risk(country_name)
        if not risk_data:
            return None
        
        context_lines = [
            f"COUNTRY RISK CONTEXT for {risk_data['country']}:",
            f"- Risk Classification: {risk_data.get('risk_class', 'Unknown')}",
            f"- Overall Risk Score: {risk_data.get('inform_risk', 'N/A')}/10",
        ]
        
        # Add key metrics that are most relevant for decision making
        key_fields = [
            ('lack_of_reliability', 'Reliability Risk'),
            ('projected_conflict_probability', 'Conflict Risk'),
            ('governance', 'Governance Risk'),
            ('infrastructure', 'Infrastructure Risk'),
            ('current_conflict_intensity', 'Security Risk')
        ]
        
        for field, label in key_fields:
            value = risk_data.get(field)
            if value is not None:
                context_lines.append(f"- {label}: {value}/10")
        
        return "\n".join(context_lines)