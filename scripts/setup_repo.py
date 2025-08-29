#!/usr/bin/env python3
"""
Repository Setup Script for Masked Critique Fine-tuning

This script helps set up the repository and verifies the installation.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False


def check_file_exists(file_path, description):
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} not found: {file_path}")
        return False


def check_directory_exists(dir_path, description):
    """Check if a directory exists."""
    if Path(dir_path).exists() and Path(dir_path).is_dir():
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ {description} not found: {dir_path}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Masked Critique Fine-tuning Repository")
    print("=" * 60)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check essential files and directories
    print("\n📋 Checking essential files and directories...")
    
    essential_files = [
        ("README.md", "Main README"),
        ("requirements.txt", "Python dependencies"),
        ("setup.py", "Package setup"),
        ("pyproject.toml", "Modern Python packaging"),
        ("Makefile", "Development commands"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose"),
        ("CONTRIBUTING.md", "Contributing guidelines"),
        ("CHANGELOG.md", "Version changelog"),
        ("LICENSE", "MIT License"),
        (".gitignore", "Git ignore rules"),
    ]
    
    essential_dirs = [
        ("btsft/", "Core training package"),
        ("docs/", "Documentation"),
        ("docs/figures/", "Research figures"),
        ("tests/", "Test suite"),
        ("examples/", "Usage examples"),
        ("data/", "Data utilities"),
        ("config/", "Training configurations"),
        (".github/workflows/", "GitHub Actions"),
    ]
    
    all_good = True
    
    for file_path, description in essential_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    for dir_path, description in essential_dirs:
        if not check_directory_exists(dir_path, description):
            all_good = False
    
    # Check data file
    if not check_file_exists("data/sanitized_data_v4.csv", "Main dataset"):
        all_good = False
    
    # Check figures
    figures = ["Fig 1.png", "figure 2.png", "Figure 3.png", "fig 4.png"]
    for figure in figures:
        if not check_file_exists(f"docs/figures/{figure}", f"Figure: {figure}"):
            all_good = False
    
    if not all_good:
        print("\n❌ Some essential files or directories are missing!")
        print("   Please ensure all files are in place before proceeding.")
        return False
    
    print("\n✅ All essential files and directories are present!")
    
    # Try to install the package
    print("\n🔧 Installing package in development mode...")
    if not run_command("pip install -e .", "Package installation"):
        print("❌ Package installation failed. Please check the error above.")
        return False
    
    # Check if the CLI command is available
    print("\n🔍 Checking CLI command availability...")
    try:
        result = subprocess.run(["mcf", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CLI command 'mcf' is available")
        else:
            print("❌ CLI command 'mcf' failed")
            all_good = False
    except FileNotFoundError:
        print("❌ CLI command 'mcf' not found")
        all_good = False
    
    # Check Python imports
    print("\n🔍 Checking Python imports...")
    try:
        import btsft
        print("✅ btsft package imported successfully")
        
        import btsft.func.training
        print("✅ Training module imported successfully")
        
        import btsft.trainers.blurred_thoughts
        print("✅ Trainer module imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        all_good = False
    
    # Run basic tests
    print("\n🧪 Running basic tests...")
    if not run_command("python -m pytest tests/ -v", "Basic tests"):
        print("⚠️  Some tests failed. This might be expected for a new setup.")
    
    # Check data loader
    print("\n📊 Testing data loader...")
    try:
        from data.data_loader import MaskedCritiqueDataLoader
        loader = MaskedCritiqueDataLoader("data/sanitized_data_v4.csv")
        df = loader.load_csv_data()
        print(f"✅ Data loader working: {len(df)} rows loaded")
    except Exception as e:
        print(f"❌ Data loader failed: {e}")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Repository setup completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Run 'make help' to see available commands")
        print("   2. Run 'make test' to run all tests")
        print("   3. Run 'make format' to format code")
        print("   4. Run 'python examples/basic_training.py' to test training")
        print("   5. Run 'python data/data_loader.py' to test data loading")
        print("\n📚 Documentation:")
        print("   - README.md: Main project documentation")
        print("   - docs/PROJECT_STRUCTURE.md: Detailed project structure")
        print("   - CONTRIBUTING.md: How to contribute")
        print("   - CHANGELOG.md: Version history")
    else:
        print("⚠️  Repository setup completed with some issues.")
        print("   Please review the errors above and fix them before proceeding.")
    
    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
