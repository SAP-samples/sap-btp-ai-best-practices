"""
Project management functionality for the RFQ Analysis Application.

This module handles saving and loading project states, including document references,
extracted features, and analysis results.
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import io

# Import country risk manager for enhanced project analysis
try:
    from country_risk_manager import CountryRiskManager
except ImportError:
    CountryRiskManager = None


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


class ProjectManager:
    """Manages project save states for the RFQ analysis application."""
    
    def __init__(self, projects_dir: str = "projects"):
        """
        Initialize the project manager.
        
        Args:
            projects_dir: Directory to store project bundles
        """
        self.projects_dir = Path(projects_dir).resolve()
        self.projects_dir.mkdir(exist_ok=True)
        
        # Initialize country risk manager if available
        self.country_risk_manager = CountryRiskManager() if CountryRiskManager else None
    
    def sanitize_project_name(self, name: str) -> str:
        """
        Sanitize project name to prevent security issues.
        
        Args:
            name: Raw project name from user input
            
        Returns:
            Sanitized project name
            
        Raises:
            ValueError: If the name is invalid
        """
        # Remove leading/trailing whitespace
        name = name.strip()
        
        # Check if empty
        if not name:
            raise ValueError("Project name cannot be empty")
        
        # Allow only alphanumeric, spaces, hyphens, and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9\s_-]', '', name)
        
        # Check if anything remains after sanitization
        if not sanitized or sanitized.isspace():
            raise ValueError("Project name must contain valid characters")
        
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized
    
    def validate_project_path(self, project_path: Path) -> None:
        """
        Validate that a project path is within the projects directory.
        
        Args:
            project_path: Path to validate
            
        Raises:
            SecurityError: If path traversal is detected
        """
        # Resolve to absolute paths
        resolved_path = project_path.resolve()
        projects_root = self.projects_dir.resolve()
        
        # Check if the resolved path is within projects directory
        try:
            resolved_path.relative_to(projects_root)
        except ValueError:
            raise SecurityError(f"Invalid project path: {project_path}")
    
    def create_project(self, name: str) -> Dict[str, Any]:
        """
        Create a new project directory structure.
        
        Args:
            name: Project name
            
        Returns:
            Project metadata
        """
        # Sanitize name
        sanitized_name = self.sanitize_project_name(name)
        project_path = self.projects_dir / sanitized_name
        
        # Validate path
        self.validate_project_path(project_path)
        
        # Check if project already exists
        if project_path.exists():
            raise ValueError(f"Project '{sanitized_name}' already exists")
        
        # Create project structure
        project_path.mkdir(parents=True)
        artifacts_path = project_path / "artifacts"
        artifacts_path.mkdir()
        
        # Create initial metadata
        metadata = {
            "project_name": sanitized_name,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Save metadata
        metadata_path = project_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def save_project(self, name: str, session_state: Dict[str, Any]) -> None:
        """
        Save current session state to a project bundle.
        
        Args:
            name: Project name
            session_state: Streamlit session state to save
        """
        # Sanitize name
        sanitized_name = self.sanitize_project_name(name)
        project_path = self.projects_dir / sanitized_name
        
        # Validate path
        self.validate_project_path(project_path)
        
        # Create project if it doesn't exist
        if not project_path.exists():
            self.create_project(sanitized_name)
        
        # Use atomic save pattern
        temp_path = project_path.with_suffix('.tmp')
        
        try:
            # Create temp directory
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir()
            
            # Create artifacts directory
            temp_artifacts = temp_path / "artifacts"
            temp_artifacts.mkdir()
            
            # Prepare state data (excluding non-serializable items)
            state_data = {
                "project_name": sanitized_name,
                "last_modified": datetime.now().isoformat(),
                "providers": session_state.get("providers", []),
                "num_providers": session_state.get("num_providers", 0),
                "custom_features": session_state.get("custom_features", []),
                "dynamic_extraction_enabled": session_state.get("dynamic_extraction_enabled", False),
                "feature_activation": session_state.get("feature_activation", {}),
                "document_features": session_state.get("document_features", {}),
                "comparison_result": session_state.get("comparison_result", {}),
                "analysis_metadata": session_state.get("analysis_metadata", {})
            }
            
            # Extract and store country risk data if available
            if self.country_risk_manager:
                country_risk_data = self._extract_country_risk_data(session_state)
                if country_risk_data:
                    state_data["country_risk_data"] = country_risk_data
            
            # Extract uploads mapping from providers
            uploads_mapping = {}
            for provider in state_data.get("providers", []):
                if provider.get("files"):
                    uploads_mapping[provider["name"]] = [f.name for f in provider["files"]]
            state_data["uploads_mapping"] = uploads_mapping
            
            # Save comparison report if exists
            if session_state.get("comparison_report"):
                report_path = temp_artifacts / "comparison_report.md"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(session_state["comparison_report"])
                state_data["has_comparison_report"] = True
            
            # Save graph image if exists
            if session_state.get("graph_image"):
                graph_path = temp_artifacts / "knowledge_graph.webp"
                if isinstance(session_state["graph_image"], (bytes, io.BytesIO)):
                    # Handle bytes or BytesIO
                    if isinstance(session_state["graph_image"], io.BytesIO):
                        graph_data = session_state["graph_image"].getvalue()
                    else:
                        graph_data = session_state["graph_image"]
                    with open(graph_path, 'wb') as f:
                        f.write(graph_data)
                    state_data["has_graph_image"] = True
            
            # Save state data
            state_path = temp_path / "state.json"
            with open(state_path, 'w', encoding='utf-8') as f:
                # Remove file objects from providers before serializing
                clean_providers = []
                for provider in state_data.get("providers", []):
                    clean_provider = provider.copy()
                    clean_provider.pop("files", None)  # Remove non-serializable file objects
                    clean_providers.append(clean_provider)
                state_data["providers"] = clean_providers
                
                json.dump(state_data, f, indent=2, default=str)
            
            # Update metadata
            metadata_path = temp_path / "metadata.json"
            metadata = {
                "project_name": sanitized_name,
                "created_at": state_data.get("created_at", datetime.now().isoformat()),
                "last_modified": datetime.now().isoformat(),
                "version": "1.0",
                "summary": {
                    "num_providers": len(state_data.get("providers", [])),
                    "has_analysis": state_data.get("has_comparison_report", False),
                    "has_graph": state_data.get("has_graph_image", False)
                }
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Atomic rename
            if project_path.exists():
                backup_path = project_path.with_suffix('.backup')
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                project_path.rename(backup_path)
            
            temp_path.rename(project_path)
            
            # Clean up backup
            backup_path = project_path.with_suffix('.backup')
            if backup_path.exists():
                shutil.rmtree(backup_path)
                
        except Exception as e:
            # Restore from backup if exists
            backup_path = project_path.with_suffix('.backup')
            if backup_path.exists() and not project_path.exists():
                backup_path.rename(project_path)
            
            # Clean up temp directory
            if temp_path.exists():
                shutil.rmtree(temp_path)
            
            raise Exception(f"Failed to save project: {str(e)}")
    
    def load_project(self, name: str) -> Dict[str, Any]:
        """
        Load a project bundle into memory.
        
        Args:
            name: Project name
            
        Returns:
            Project state data
        """
        # Sanitize name
        sanitized_name = self.sanitize_project_name(name)
        project_path = self.projects_dir / sanitized_name
        
        # Validate path
        self.validate_project_path(project_path)
        
        # Check if project exists
        if not project_path.exists():
            raise ValueError(f"Project '{sanitized_name}' not found")
        
        # Load state data
        state_path = project_path / "state.json"
        if not state_path.exists():
            raise ValueError(f"Project state file not found")
        
        with open(state_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # Load artifacts if they exist
        artifacts_path = project_path / "artifacts"
        
        # Load comparison report
        report_path = artifacts_path / "comparison_report.md"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                state_data["comparison_report"] = f.read()
        
        # Load graph image
        graph_path = artifacts_path / "knowledge_graph.webp"
        if graph_path.exists():
            with open(graph_path, 'rb') as f:
                state_data["graph_image"] = io.BytesIO(f.read())
        
        return state_data
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all available projects.
        
        Returns:
            List of project metadata
        """
        projects = []
        
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                metadata_path = project_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        projects.append(metadata)
                    except Exception:
                        # Skip corrupted projects
                        continue
        
        # Sort by last modified date
        projects.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
        
        return projects
    
    def get_project_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a project.
        
        Args:
            name: Project name
            
        Returns:
            Project information including metadata and state summary
        """
        # Load project state
        state_data = self.load_project(name)
        
        # Prepare project info
        info = {
            "project_name": state_data.get("project_name", name),
            "created_at": state_data.get("created_at", "Unknown"),
            "last_modified": state_data.get("last_modified", "Unknown"),
            "providers": [],
            "has_analysis": state_data.get("has_comparison_report", False),
            "has_graph": state_data.get("has_graph_image", False),
            "uploads_mapping": state_data.get("uploads_mapping", {})
        }
        
        # Extract provider information
        for provider in state_data.get("providers", []):
            provider_info = {
                "name": provider.get("name", "Unknown"),
                "status": provider.get("status", "unknown"),
                "file_count": len(state_data.get("uploads_mapping", {}).get(provider.get("name", ""), [])),
                "files": state_data.get("uploads_mapping", {}).get(provider.get("name", ""), [])
            }
            info["providers"].append(provider_info)
        
        return info
    
    def delete_project(self, name: str) -> None:
        """
        Delete a project (does not delete uploaded files).
        
        Args:
            name: Project name
        """
        # Sanitize name
        sanitized_name = self.sanitize_project_name(name)
        project_path = self.projects_dir / sanitized_name
        
        # Validate path
        self.validate_project_path(project_path)
        
        # Check if project exists
        if not project_path.exists():
            raise ValueError(f"Project '{sanitized_name}' not found")
        
        # Remove project directory
        shutil.rmtree(project_path)
    
    def project_exists(self, name: str) -> bool:
        """
        Check if a project exists.
        
        Args:
            name: Project name
            
        Returns:
            True if project exists
        """
        try:
            sanitized_name = self.sanitize_project_name(name)
            project_path = self.projects_dir / sanitized_name
            return project_path.exists()
        except ValueError:
            return False
    
    def _extract_country_risk_data(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract country risk data from the current session state.
        
        Args:
            session_state: Streamlit session state
            
        Returns:
            Dictionary containing country risk data for all found countries
        """
        if not self.country_risk_manager:
            return {}
        
        country_risk_data = {}
        
        # Extract countries from provider data
        providers = session_state.get("providers", [])
        for provider in providers:
            if provider.get("extracted_data"):
                extracted_data = provider["extracted_data"]
                project_info = extracted_data.get("project_information", {})
                country = project_info.get("country_of_contracting_authority")
                
                if country and country != "Not Found":
                    # Clean country name
                    country = country.strip()
                    
                    # Get risk data for this country
                    risk_data = self.country_risk_manager.find_country_risk(country)
                    if risk_data:
                        country_risk_data[country] = {
                            "risk_data": risk_data,
                            "risk_summary": self.country_risk_manager.get_risk_summary(country),
                            "detailed_analysis": self.country_risk_manager.get_detailed_risk_analysis(country),
                            "provider_name": provider.get("name", "Unknown")
                        }
        
        return country_risk_data