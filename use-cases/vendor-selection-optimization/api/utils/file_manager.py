"""File Management Utilities for API operations"""

import os
import csv
import gzip
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """Handles file operations for the API"""
    
    def __init__(self):
        self.temp_dir = settings.TEMP_DIR
        self.results_dir = settings.RESULTS_DIR
        self.profiles_dir = settings.PROFILES_DIR
        self.tables_dir = settings.TABLES_DIR
        
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_profile_path(self, profile_id: str) -> Path:
        """Get profile directory path"""
        return self.profiles_dir / profile_id
    
    def get_profile_tables_dir(self, profile_id: str) -> Path:
        """Get profile-specific tables directory"""
        return self.get_profile_path(profile_id) / "tables"
    
    def get_profile_config_dir(self, profile_id: str) -> Path:
        """Get profile-specific config directory"""
        return self.get_profile_path(profile_id) / "config"
    
    def profile_exists(self, profile_id: str) -> bool:
        """Check if profile exists"""
        profile_path = self.get_profile_path(profile_id)
        return profile_path.exists() and profile_path.is_dir()
    
    def get_temp_file_path(self, filename: str) -> Path:
        """Get a temp file path"""
        return self.temp_dir / filename
    
    def get_result_file_path(self, job_id: str, filename: str) -> Path:
        """Get a result file path for a job"""
        job_dir = self.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir / filename
    
    def estimate_csv_size(self, rows: int, columns: int, avg_cell_size: int = 20) -> int:
        """Estimate CSV file size in bytes"""
        # Account for headers, delimiters, and newlines
        header_size = columns * avg_cell_size
        row_size = columns * (avg_cell_size + 1)  # +1 for delimiter
        total_size = header_size + (rows * row_size)
        return total_size
    
    def count_csv_rows(self, file_path: Path) -> int:
        """Count rows in a CSV file efficiently"""
        if not file_path.exists():
            return 0
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
            return max(0, row_count)
        except Exception as e:
            logger.error(f"Error counting rows in {file_path}: {e}")
            return 0
    
    def read_csv_headers(self, file_path: Path) -> List[str]:
        """Read CSV headers"""
        if not file_path.exists():
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
            return headers
        except Exception as e:
            logger.error(f"Error reading headers from {file_path}: {e}")
            return []
    
    def save_dataframe_to_csv(
        self,
        df: pd.DataFrame,
        file_path: Path,
        compress: bool = False
    ) -> Tuple[Path, int]:
        """Save DataFrame to CSV, optionally compressed"""
        try:
            if compress:
                # Save as gzipped CSV
                gz_path = Path(str(file_path) + '.gz')
                df.to_csv(gz_path, index=False, compression='gzip')
                file_size = gz_path.stat().st_size
                return gz_path, file_size
            else:
                df.to_csv(file_path, index=False)
                file_size = file_path.stat().st_size
                return file_path, file_size
        except Exception as e:
            logger.error(f"Error saving dataframe to {file_path}: {e}")
            raise
    
    def clean_temp_files(self, pattern: str = "*") -> int:
        """Clean temporary files matching pattern"""
        count = 0
        try:
            for file_path in self.temp_dir.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    count += 1
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
        return count
    
    def clean_job_results(self, job_id: str) -> bool:
        """Clean all results for a job"""
        job_dir = self.results_dir / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                return True
            except Exception as e:
                logger.error(f"Error cleaning job results for {job_id}: {e}")
        return False
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information"""
        if not file_path.exists():
            return {"exists": False}
            
        stat = file_path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "is_compressed": file_path.suffix == '.gz'
        }
    
    def copy_profile_file(
        self,
        source_profile: str,
        target_profile: str,
        filename: str,
        subfolder: str = "tables"
    ) -> bool:
        """Copy a file between profiles"""
        source = self.get_profile_path(source_profile) / subfolder / filename
        target = self.get_profile_path(target_profile) / subfolder / filename
        
        if not source.exists():
            logger.error(f"Source file does not exist: {source}")
            return False
            
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return True
        except Exception as e:
            logger.error(f"Error copying file from {source} to {target}: {e}")
            return False
    
    def delete_profile_file(
        self,
        profile_id: str,
        filename: str,
        subfolder: str = "tables"
    ) -> bool:
        """Delete a file from a profile"""
        file_path = self.get_profile_path(profile_id) / subfolder / filename
        
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        return False
    
    def list_profile_files(
        self,
        profile_id: str,
        subfolder: str = "tables",
        pattern: str = "*"
    ) -> List[Dict[str, Any]]:
        """List files in a profile subfolder"""
        folder_path = self.get_profile_path(profile_id) / subfolder
        
        if not folder_path.exists():
            return []
            
        files = []
        for file_path in folder_path.glob(pattern):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    **self.get_file_info(file_path)
                })
                
        return sorted(files, key=lambda x: x["name"])
    
    def create_results_archive(self, job_id: str, file_patterns: List[str]) -> Optional[Path]:
        """Create a ZIP archive of result files for a job"""
        import zipfile
        
        job_dir = self.results_dir / job_id
        if not job_dir.exists():
            return None
        
        archive_path = job_dir / f"results_{job_id}.zip"
        
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for pattern in file_patterns:
                    for file_path in job_dir.glob(pattern):
                        if file_path.is_file() and file_path.suffix != '.zip':
                            zip_file.write(file_path, file_path.name)
                            logger.info(f"Added {file_path.name} to archive")
            
            return archive_path if archive_path.stat().st_size > 0 else None
            
        except Exception as e:
            logger.error(f"Error creating archive for job {job_id}: {e}")
            if archive_path.exists():
                archive_path.unlink()
            return None


# Global file manager instance
file_manager = FileManager()