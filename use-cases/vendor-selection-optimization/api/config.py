"""API Configuration Module"""

import os
from pathlib import Path

class Settings:
    """API Settings and Configuration"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent  # resources directory
    API_DIR = Path(__file__).parent
    PROFILES_DIR = BASE_DIR / "profiles"
    TABLES_DIR = BASE_DIR / "tables"
    CONFIG_DIR = BASE_DIR / "config"
    OPTIMIZATION_DIR = BASE_DIR / "optimization"
    
    # API Settings
    API_TITLE = "Procurement Assistant API"
    API_VERSION = "0.1.0"
    API_DESCRIPTION = """
    Procurement optimization API that provides vendor evaluation, 
    procurement allocation optimization, and policy comparison capabilities.
    """
    
    # Request/Response Settings
    MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_INLINE_RECORDS = 1000
    MAX_INLINE_SIZE_MB = 1.0
    
    # Job Management Settings
    JOB_STORAGE_TYPE = os.getenv("JOB_STORAGE_TYPE", "filesystem")  # filesystem or redis
    JOB_STORAGE_PATH = API_DIR / "job_storage"
    JOB_TTL_HOURS = int(os.getenv("JOB_TTL_HOURS", "168"))  # 7 days default
    
    # Background Task Settings
    ENABLE_BACKGROUND_TASKS = os.getenv("ENABLE_BACKGROUND_TASKS", "true").lower() == "true"
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
    
    # CORS Settings
    CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Management
    TEMP_DIR = API_DIR / "temp"
    RESULTS_DIR = API_DIR / "results"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        cls.JOB_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()