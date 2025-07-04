#!/usr/bin/env python3
"""
Simple setup checker that works without dependencies.
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("Ijon PDF RAG System - Setup Check")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version}")

# Check if .env exists
env_path = Path(".env")
if env_path.exists():
    print(f"✓ .env file found at: {env_path.absolute()}")
    
    # Check for key environment variables
    with open(env_path) as f:
        env_content = f.read()
        
    keys_to_check = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT",
        "VECTOR_DB_TYPE",
    ]
    
    print("\nEnvironment variables:")
    for key in keys_to_check:
        if key in env_content:
            print(f"  ✓ {key} is set")
        else:
            print(f"  ✗ {key} is missing")
else:
    print("✗ No .env file found")
    print("  You should copy .env.example to .env and add your API keys")

# Check for required directories
print("\nDirectories:")
dirs_to_check = ["src", "tests", "utils", "sample_pdfs", "logs"]
for dir_name in dirs_to_check:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"  ✓ {dir_name}/ exists")
    else:
        print(f"  ✗ {dir_name}/ missing (will be created)")

# Check for sample PDFs
sample_dir = Path("sample_pdfs")
if sample_dir.exists():
    pdfs = list(sample_dir.glob("*.pdf"))
    if pdfs:
        print(f"\n✓ Found {len(pdfs)} sample PDFs:")
        for pdf in pdfs:
            print(f"  - {pdf.name}")
    else:
        print("\n✗ No sample PDFs found")
        print("  Run: python3 utils/generate_sample_pdfs.py")
else:
    print("\n✗ sample_pdfs/ directory doesn't exist")

# Check for key Python packages
print("\nChecking Python packages (basic import test):")
packages = {
    "pydantic": "Core data validation",
    "reportlab": "PDF generation",
    "rich": "Terminal UI",
}

for package, description in packages.items():
    try:
        __import__(package)
        print(f"  ✓ {package} - {description}")
    except ImportError:
        print(f"  ✗ {package} - {description} (not installed)")

# Next steps
print("\n" + "=" * 60)
print("Next Steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Set up .env file with your API keys")
print("3. Generate sample PDFs: python3 utils/generate_sample_pdfs.py")
print("4. Run full initialization: python3 initialize_system.py")
print("5. Test the system: python3 test_system.py")
print("=" * 60)