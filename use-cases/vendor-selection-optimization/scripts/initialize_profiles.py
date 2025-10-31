#!/usr/bin/env python3
"""
Initialize Optimization Profiles System

This script sets up the directory structure and default configurations
for all optimization profiles.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the resources directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimization.profile_manager import ProfileManager


def main():
    """Initialize all optimization profiles."""
    print("Initializing Optimization Profiles System...")
    
    # Get the base path (resources directory)
    base_path = Path(__file__).parent.parent
    
    # Initialize ProfileManager
    profile_manager = ProfileManager(str(base_path))
    
    print("Setting up profile directory structure...")
    
    # Initialize all profiles
    profile_manager.initialize_all_profiles()
    
    print("Profile initialization completed!")
    print("\nProfile structure created:")
    
    # Display the created structure
    for profile_id in ["profile_1", "profile_2", "profile_3", "profile_4", "profile_5"]:
        profile_dir = profile_manager.get_profile_directory(profile_id)
        if os.path.exists(profile_dir):
            print(f"  ✓ {profile_id}/")
            print(f"    - config/")
            print(f"    - tables/")
            print(f"    - profile_metadata.json")
    
    # Display profile information
    profiles = profile_manager.get_profile_list()
    print("\nConfigured Profiles:")
    for profile in profiles:
        active_marker = " (ACTIVE)" if profile["is_active"] else ""
        print(f"  {profile['profile_id']}: {profile['name']}{active_marker}")
    
    print("\nOptimization Profiles System is ready!")


if __name__ == "__main__":
    main()