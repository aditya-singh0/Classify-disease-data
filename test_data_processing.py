#!/usr/bin/env python3
"""
Test Data Processing
Quick test to verify data loading and processing works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing import DatasetProcessor, AbstractProcessor

def test_data_processing():
    """Test data processing pipeline."""
    print("🧪 Testing Data Processing Pipeline")
    print("=" * 50)
    
    try:
        # Initialize processors
        abstract_processor = AbstractProcessor()
        dataset_processor = DatasetProcessor(abstract_processor)
        
        print("✅ Processors initialized")
        
        # Check if dataset folders exist
        cancer_dir = Path("Dataset/Cancer")
        non_cancer_dir = Path("Dataset/Non-Cancer")
        
        if not cancer_dir.exists():
            print(f"❌ Cancer directory not found: {cancer_dir}")
            return False
            
        if not non_cancer_dir.exists():
            print(f"❌ Non-Cancer directory not found: {non_cancer_dir}")
            return False
        
        print(f"✅ Dataset directories found")
        print(f"   Cancer files: {len(list(cancer_dir.glob('*.txt')))}")
        print(f"   Non-Cancer files: {len(list(non_cancer_dir.glob('*.txt')))}")
        
        # Load a few sample files to test processing
        cancer_files = list(cancer_dir.glob('*.txt'))[:5]
        non_cancer_files = list(non_cancer_dir.glob('*.txt'))[:5]
        
        print(f"\n📖 Testing file parsing...")
        
        # Test cancer files
        for file_path in cancer_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.strip().split('\n')
                if len(lines) >= 3:
                    pmid = file_path.stem
                    title = lines[1] if len(lines) > 1 else ""
                    abstract = lines[2] if len(lines) > 2 else ""
                    
                    # Test cleaning
                    cleaned_abstract = abstract_processor.clean_abstract(abstract)
                    diseases = abstract_processor.extract_diseases(abstract)
                    
                    print(f"   ✅ {pmid}: {len(cleaned_abstract)} chars, {len(diseases)} diseases")
                else:
                    print(f"   ⚠️  {file_path.name}: Unexpected format")
                    
            except Exception as e:
                print(f"   ❌ {file_path.name}: {e}")
        
        # Test non-cancer files
        for file_path in non_cancer_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.strip().split('\n')
                if len(lines) >= 3:
                    pmid = file_path.stem
                    title = lines[1] if len(lines) > 1 else ""
                    abstract = lines[2] if len(lines) > 2 else ""
                    
                    # Test cleaning
                    cleaned_abstract = abstract_processor.clean_abstract(abstract)
                    diseases = abstract_processor.extract_diseases(abstract)
                    
                    print(f"   ✅ {pmid}: {len(cleaned_abstract)} chars, {len(diseases)} diseases")
                else:
                    print(f"   ⚠️  {file_path.name}: Unexpected format")
                    
            except Exception as e:
                print(f"   ❌ {file_path.name}: {e}")
        
        print(f"\n✅ Data processing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_data_processing()
    sys.exit(0 if success else 1) 