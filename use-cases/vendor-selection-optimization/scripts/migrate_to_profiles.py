#!/usr/bin/env python3
"""
Migration Script for Optimization Profiles System

This script migrates existing optimization data to the new profile system.
It moves generated CSV files to Profile 1 and creates backups of the original files.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add the resources directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimization.profile_manager import ProfileManager


def backup_files(source_dir: str, backup_dir: str, files_to_backup: list) -> None:
    """Create backups of files before migration."""
    os.makedirs(backup_dir, exist_ok=True)
    
    for file_name in files_to_backup:
        source_path = os.path.join(source_dir, file_name)
        if os.path.exists(source_path):
            backup_path = os.path.join(backup_dir, file_name)
            shutil.copy2(source_path, backup_path)
            print(f"  Backed up: {file_name}")


def migrate_generated_files(source_dir: str, profile_manager: ProfileManager) -> None:
    """Migrate existing generated files to Profile 1."""
    target_profile = "profile_1"
    target_dir = profile_manager.get_profile_tables_dir(target_profile)
    
    # List of generated files to migrate
    files_to_migrate = profile_manager.get_generated_files()
    
    print(f"Migrating generated files to {target_profile}...")
    
    for file_name in files_to_migrate:
        source_path = os.path.join(source_dir, file_name)
        if os.path.exists(source_path):
            target_path = os.path.join(target_dir, file_name)
            shutil.move(source_path, target_path)
            print(f"  Migrated: {file_name}")
        else:
            print(f"  Skipped (not found): {file_name}")


def main():
    """Main migration function."""
    print("Starting migration to Optimization Profiles System...")
    
    # Get the base path (resources directory)
    base_path = Path(__file__).parent.parent
    
    # Initialize ProfileManager
    profile_manager = ProfileManager(str(base_path))
    
    # Current tables directory
    current_tables_dir = profile_manager.get_global_tables_dir()
    
    # Create backup directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(current_tables_dir, "backup", f"pre_migration_{timestamp}")
    
    # List of files to migrate
    files_to_migrate = profile_manager.get_generated_files()
    
    print(f"Current tables directory: {current_tables_dir}")
    print(f"Backup directory: {backup_dir}")
    print(f"Files to migrate: {files_to_migrate}")
    
    # Check if any files exist to migrate
    existing_files = [f for f in files_to_migrate if os.path.exists(os.path.join(current_tables_dir, f))]
    
    if not existing_files:
        print("No generated files found to migrate.")
        print("Migration completed successfully (no files to migrate).")
        return
    
    print(f"Found {len(existing_files)} files to migrate: {existing_files}")
    
    # Create backup of existing files
    print("\nCreating backup of existing files...")
    backup_files(current_tables_dir, backup_dir, existing_files)
    
    # Migrate files to Profile 1
    print("\nMigrating files to Profile 1...")
    migrate_generated_files(current_tables_dir, profile_manager)
    
    # Verify migration
    print("\nVerifying migration...")
    profile_1_tables_dir = profile_manager.get_profile_tables_dir("profile_1")
    
    for file_name in existing_files:
        target_path = os.path.join(profile_1_tables_dir, file_name)
        if os.path.exists(target_path):
            print(f"  ✓ {file_name} migrated successfully")
        else:
            print(f"  ✗ {file_name} migration failed")
    
    print("\nMigration to Profile System completed!")
    print("\nNext steps:")
    print("1. The existing optimization results are now in Profile 1")
    print("2. SAP master data remains in the global tables directory")
    print("3. Configuration files have been copied to all profiles")
    print("4. You can now use the Profile selection interface in Settings")


if __name__ == "__main__":
    main()