"""
Profile Management System for Optimization Profiles.

This module provides functionality to manage multiple optimization profiles,
each with its own configuration settings and data directories.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import streamlit as st


class ProfileManager:
    """Manages optimization profiles with separate configurations and data directories."""
    
    def __init__(self, base_path: str = "resources"):
        """
        Initialize the ProfileManager.
        
        Args:
            base_path: Base path for the application resources
        """
        self.base_path = base_path
        self.profiles_dir = os.path.join(base_path, "profiles")
        self.profiles_config_file = os.path.join(self.profiles_dir, "profiles_config.json")
        self.current_config_dir = os.path.join(base_path, "config")
        self.current_tables_dir = os.path.join(base_path, "tables")
        
        # Ensure profiles directory exists
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Initialize profiles configuration if it doesn't exist
        self._initialize_profiles_config()
    
    def _initialize_profiles_config(self) -> None:
        """Initialize the profiles configuration file if it doesn't exist."""
        if not os.path.exists(self.profiles_config_file):
            default_config = {
                "active_profile": "profile_1",
                "profiles": {
                    "profile_1": {
                        "name": "Default Configuration",
                        "description": "Default optimization configuration",
                        "created_date": datetime.now().isoformat(),
                        "last_modified": datetime.now().isoformat()
                    },
                    "profile_2": {
                        "name": "Configuration 2",
                        "description": "Alternative optimization configuration",
                        "created_date": datetime.now().isoformat(),
                        "last_modified": datetime.now().isoformat()
                    },
                    "profile_3": {
                        "name": "Configuration 3",
                        "description": "Alternative optimization configuration",
                        "created_date": datetime.now().isoformat(),
                        "last_modified": datetime.now().isoformat()
                    },
                    "profile_4": {
                        "name": "Configuration 4",
                        "description": "Alternative optimization configuration",
                        "created_date": datetime.now().isoformat(),
                        "last_modified": datetime.now().isoformat()
                    },
                    "profile_5": {
                        "name": "Configuration 5",
                        "description": "Alternative optimization configuration",
                        "created_date": datetime.now().isoformat(),
                        "last_modified": datetime.now().isoformat()
                    }
                }
            }
            self._save_profiles_config(default_config)
    
    def _save_profiles_config(self, config: Dict) -> None:
        """Save the profiles configuration to file."""
        with open(self.profiles_config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_profiles_config(self) -> Dict:
        """Load the profiles configuration from file."""
        try:
            with open(self.profiles_config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._initialize_profiles_config()
            return self._load_profiles_config()
    
    def get_profile_directory(self, profile_id: str) -> str:
        """Get the directory path for a specific profile."""
        return os.path.join(self.profiles_dir, profile_id)
    
    def get_profile_config_dir(self, profile_id: str) -> str:
        """Get the config directory path for a specific profile."""
        return os.path.join(self.get_profile_directory(profile_id), "config")
    
    def get_profile_tables_dir(self, profile_id: str) -> str:
        """Get the tables directory path for a specific profile."""
        return os.path.join(self.get_profile_directory(profile_id), "tables")
    
    def create_profile_structure(self, profile_id: str) -> None:
        """Create the directory structure for a profile."""
        profile_dir = self.get_profile_directory(profile_id)
        config_dir = self.get_profile_config_dir(profile_id)
        tables_dir = self.get_profile_tables_dir(profile_id)
        
        # Create directories
        os.makedirs(profile_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        
        # Create profile metadata file
        metadata_file = os.path.join(profile_dir, "profile_metadata.json")
        if not os.path.exists(metadata_file):
            profiles_config = self._load_profiles_config()
            profile_info = profiles_config["profiles"].get(profile_id, {})
            
            metadata = {
                "profile_id": profile_id,
                "name": profile_info.get("name", f"Profile {profile_id}"),
                "description": profile_info.get("description", ""),
                "created_date": profile_info.get("created_date", datetime.now().isoformat()),
                "last_modified": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def copy_current_config_to_profile(self, profile_id: str) -> None:
        """Copy current configuration files to a profile directory."""
        profile_config_dir = self.get_profile_config_dir(profile_id)
        profile_tables_dir = self.get_profile_tables_dir(profile_id)
        
        # Copy config files
        if os.path.exists(self.current_config_dir):
            for file_name in os.listdir(self.current_config_dir):
                if file_name.endswith('.json'):
                    src = os.path.join(self.current_config_dir, file_name)
                    dst = os.path.join(profile_config_dir, file_name)
                    shutil.copy2(src, dst)
        
        # Copy only profile-specific configuration files (not SAP master data)
        profile_specific_files = [
            "tariff_values.json",
            "logistics_factors.json"
        ]
        
        if os.path.exists(self.current_tables_dir):
            for file_name in profile_specific_files:
                src = os.path.join(self.current_tables_dir, file_name)
                if os.path.exists(src):
                    dst = os.path.join(profile_tables_dir, file_name)
                    shutil.copy2(src, dst)
    
    def load_profile_config(self, profile_id: str) -> Tuple[Dict, Dict]:
        """
        Load configuration from a profile.
        
        Returns:
            Tuple of (costs_config, tariff_config)
        """
        profile_config_dir = self.get_profile_config_dir(profile_id)
        profile_tables_dir = self.get_profile_tables_dir(profile_id)
        
        # Load costs configuration
        costs_config = {}
        costs_file = os.path.join(profile_config_dir, "costs.json")
        if os.path.exists(costs_file):
            with open(costs_file, 'r') as f:
                costs_config = json.load(f)
        
        # Load tariff configuration
        tariff_config = {}
        tariff_file = os.path.join(profile_tables_dir, "tariff_values.json")
        if os.path.exists(tariff_file):
            with open(tariff_file, 'r') as f:
                tariff_config = json.load(f)
        
        return costs_config, tariff_config
    
    def save_profile_config(self, profile_id: str, costs_config: Dict, tariff_config: Dict) -> None:
        """
        Save configuration to a profile.
        
        Args:
            profile_id: Profile identifier
            costs_config: Economic Impact Parameters configuration
            tariff_config: Tariff values configuration
        """
        profile_config_dir = self.get_profile_config_dir(profile_id)
        profile_tables_dir = self.get_profile_tables_dir(profile_id)
        
        # Ensure directories exist
        os.makedirs(profile_config_dir, exist_ok=True)
        os.makedirs(profile_tables_dir, exist_ok=True)
        
        # Save costs configuration
        costs_file = os.path.join(profile_config_dir, "costs.json")
        with open(costs_file, 'w') as f:
            json.dump(costs_config, f, indent=2)
        
        # Save tariff configuration
        tariff_file = os.path.join(profile_tables_dir, "tariff_values.json")
        with open(tariff_file, 'w') as f:
            json.dump(tariff_config, f, indent=2)
        
        # Update profile metadata
        self._update_profile_metadata(profile_id)
    
    def _update_profile_metadata(self, profile_id: str) -> None:
        """Update the last modified timestamp for a profile."""
        profile_dir = self.get_profile_directory(profile_id)
        metadata_file = os.path.join(profile_dir, "profile_metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata["last_modified"] = datetime.now().isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def get_active_profile(self) -> str:
        """Get the currently active profile ID."""
        profiles_config = self._load_profiles_config()
        return profiles_config.get("active_profile", "profile_1")
    
    def set_active_profile(self, profile_id: str) -> None:
        """Set the active profile."""
        profiles_config = self._load_profiles_config()
        profiles_config["active_profile"] = profile_id
        self._save_profiles_config(profiles_config)
        
        # Update session state if running in Streamlit
        if hasattr(st, 'session_state'):
            st.session_state['active_profile'] = profile_id
    
    def get_profile_list(self) -> List[Dict]:
        """Get a list of all available profiles with their metadata."""
        profiles_config = self._load_profiles_config()
        profile_list = []
        
        for profile_id, profile_info in profiles_config["profiles"].items():
            metadata = {
                "profile_id": profile_id,
                "name": profile_info.get("name", f"Profile {profile_id}"),
                "description": profile_info.get("description", ""),
                "created_date": profile_info.get("created_date", ""),
                "last_modified": profile_info.get("last_modified", ""),
                "is_active": profile_id == self.get_active_profile()
            }
            profile_list.append(metadata)
        
        return profile_list
    
    def update_profile_name(self, profile_id: str, new_name: str) -> None:
        """Update the name of a profile."""
        profiles_config = self._load_profiles_config()
        if profile_id in profiles_config["profiles"]:
            profiles_config["profiles"][profile_id]["name"] = new_name
            profiles_config["profiles"][profile_id]["last_modified"] = datetime.now().isoformat()
            self._save_profiles_config(profiles_config)
    
    def update_profile_description(self, profile_id: str, new_description: str) -> None:
        """Update the description of a profile."""
        profiles_config = self._load_profiles_config()
        if profile_id in profiles_config["profiles"]:
            profiles_config["profiles"][profile_id]["description"] = new_description
            profiles_config["profiles"][profile_id]["last_modified"] = datetime.now().isoformat()
            self._save_profiles_config(profiles_config)
    
    def initialize_all_profiles(self) -> None:
        """Initialize all profiles with current configuration."""
        profiles_config = self._load_profiles_config()
        
        for profile_id in profiles_config["profiles"].keys():
            # Create profile structure
            self.create_profile_structure(profile_id)
            
            # Copy current configuration to profile if it doesn't exist
            profile_config_dir = self.get_profile_config_dir(profile_id)
            costs_file = os.path.join(profile_config_dir, "costs.json")
            
            if not os.path.exists(costs_file):
                self.copy_current_config_to_profile(profile_id)
    
    def get_profile_paths(self, profile_id: str) -> Dict[str, str]:
        """Get all important paths for a profile."""
        return {
            "profile_dir": self.get_profile_directory(profile_id),
            "config_dir": self.get_profile_config_dir(profile_id),
            "tables_dir": self.get_profile_tables_dir(profile_id),
            "costs_file": os.path.join(self.get_profile_config_dir(profile_id), "costs.json"),
            "tariff_file": os.path.join(self.get_profile_tables_dir(profile_id), "tariff_values.json")
        }
    
    def profile_exists(self, profile_id: str) -> bool:
        """Check if a profile exists and is properly configured."""
        profile_paths = self.get_profile_paths(profile_id)
        return (
            os.path.exists(profile_paths["profile_dir"]) and
            os.path.exists(profile_paths["config_dir"]) and
            os.path.exists(profile_paths["tables_dir"])
        )
    
    def get_global_tables_dir(self) -> str:
        """Get the global tables directory where SAP master data is stored."""
        return self.current_tables_dir
    
    def get_sap_table_files(self) -> List[str]:
        """Get list of SAP master data files that should remain global."""
        return [
            "SAP_VLY_IL_VPI_TARGET_VALUES.csv",
            "SAP_VLY_IL_SUPPLIER_GROUP.csv",
            "SAP_VLY_IL_SUPPLIER_GEO.csv",
            "SAP_VLY_IL_SUPPLIER.csv",
            "SAP_VLY_IL_PURCHASE_ORG.csv",
            "SAP_VLY_IL_PURCHASE_GROUP.csv",
            "SAP_VLY_IL_PO_SCHEDULE_LINES.csv",
            "SAP_VLY_IL_PO_ITEMS.csv",
            "SAP_VLY_IL_PO_HISTORY.csv",
            "SAP_VLY_IL_PO_HEADER.csv",
            "SAP_VLY_IL_ORDERING_PLANT_GEO.csv",
            "SAP_VLY_IL_ORDERING_PLANT.csv",
            "SAP_VLY_IL_MATERIAL_GROUP.csv",
            "SAP_VLY_IL_MATERIAL.csv",
            "SAP_VLY_IL_LEADTIME_INDEX.csv",
            "SAP_VLY_IL_DELIVERY_TOLERANCES.csv",
            "SAP_VLY_IL_COUNTRY.csv",
            "SAP_VLY_HL_BASE_PO_STATUS_$PT1.csv",
            "SAP_VLY_HL_BASE_LAYER_OTIF_$PT1.csv",
            "SAP_VLY_HL_BASE_LAYER_LEADTIME_$PT2.csv"
        ]
    
    def get_profile_specific_files(self) -> List[str]:
        """Get list of profile-specific configuration files."""
        return [
            "tariff_values.json",
            "logistics_factors.json"
        ]
    
    def get_generated_files(self) -> List[str]:
        """Get list of files that are generated by optimization and should be cleaned."""
        return [
            "vendor_matnr_ranking_tariff_values.csv",
            "vendor_with_direct_countries.csv",
            "optimized_allocation_matnr_vendor_matnr_ranking_tariff_values.csv",
            "comparison.csv"
        ]
    
    def clean_profile_data(self, profile_id: str) -> None:
        """Clean generated data files for a profile to force regeneration."""
        profile_tables_dir = self.get_profile_tables_dir(profile_id)
        
        # Only clean generated files, not SAP master data or configuration files
        files_to_clean = self.get_generated_files()
        
        for file_name in files_to_clean:
            file_path = os.path.join(profile_tables_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def get_data_file_path(self, profile_id: str, file_name: str) -> str:
        """
        Get the appropriate path for a data file based on whether it's global or profile-specific.
        
        Args:
            profile_id: Profile identifier
            file_name: Name of the file
            
        Returns:
            Full path to the file (either global or profile-specific)
        """
        sap_files = self.get_sap_table_files()
        
        if file_name in sap_files:
            # SAP master data files are global
            return os.path.join(self.get_global_tables_dir(), file_name)
        else:
            # Profile-specific files (configurations and generated files)
            return os.path.join(self.get_profile_tables_dir(profile_id), file_name)