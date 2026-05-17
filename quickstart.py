"""
Quick Start Script - Get started in 3 commands

This script helps you quickly set up and run the investigation
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("FACIAL RECOGNITION INVESTIGATION - QUICK START")
    print("="*60)
    
    # Step 1: Install dependencies
    print("\nStep 1: Installing Python dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Dependencies"):
        print("\n✗ Failed to install dependencies")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✓ Dependencies installed successfully")
    
    # Step 2: Run main application
    print("\nStep 2: Launching Facial Recognition Investigation...")
    print("\nThe application will guide you through:")
    print("  1. Setting up the face database")
    print("  2. Running preprocessing techniques tests")
    print("  3. Running face recognition models tests")
    print("  4. Performing statistical analysis")
    print("  5. Viewing detailed results")
    
    input("\nPress Enter to continue...")
    
    # Step 3: Launch main application
    try:
        import main
        main.main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTry running: python main.py")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
