#!/usr/bin/env python3
"""
Data Loading Utilities for Masked Critique Fine-tuning

This module provides utilities for loading and processing training data.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import Dataset


class MaskedCritiqueDataLoader:
    """Data loader for Masked Critique Fine-tuning datasets."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data file (CSV, JSONL, etc.)
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_csv_data(self) -> pd.DataFrame:
        """
        Load CSV data and perform basic validation.
        
        Returns:
            Loaded and validated DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"📊 Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Basic validation
        required_columns = ['prompt', 'critique']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter valid examples
        if 'answer_match' in df.columns:
            df = df[df['answer_match'] == True]
            print(f"✅ Filtered to {len(df)} valid examples")
        
        self.data = df
        return df
    
    def convert_to_huggingface_dataset(self) -> Dataset:
        """
        Convert the loaded data to a HuggingFace Dataset.
        
        Returns:
            HuggingFace Dataset object
        """
        if self.data is None:
            self.load_csv_data()
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(self.data)
        print(f"🔄 Converted to HuggingFace Dataset with {len(dataset)} examples")
        
        return dataset
    
    def get_sample_data(self, n_samples: int = 5) -> List[Dict]:
        """
        Get sample data for inspection.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            List of sample data dictionaries
        """
        if self.data is None:
            self.load_csv_data()
        
        samples = self.data.head(n_samples).to_dict('records')
        return samples
    
    def validate_data_format(self) -> bool:
        """
        Validate that the data follows the expected format.
        
        Returns:
            True if data format is valid
        """
        if self.data is None:
            self.load_csv_data()
        
        # Check for required structure
        required_structure = {
            'prompt': 'User question/prompt',
            'critique': 'Model critique and reasoning'
        }
        
        print("🔍 Validating data format...")
        
        for column, description in required_structure.items():
            if column not in self.data.columns:
                print(f"❌ Missing column: {column} ({description})")
                return False
        
        # Check for non-empty values
        for column in required_structure.keys():
            empty_count = self.data[column].isna().sum()
            if empty_count > 0:
                print(f"⚠️  Column '{column}' has {empty_count} empty values")
        
        print("✅ Data format validation completed")
        return True


def load_sanitized_dataset(data_path: str = "data/sanitized_data_v4.csv") -> Dataset:
    """
    Convenience function to load the sanitized dataset.
    
    Args:
        data_path: Path to the sanitized data file
        
    Returns:
        HuggingFace Dataset object
    """
    loader = MaskedCritiqueDataLoader(data_path)
    return loader.convert_to_huggingface_dataset()


if __name__ == "__main__":
    # Example usage
    try:
        dataset = load_sanitized_dataset()
        print(f"📚 Dataset loaded successfully with {len(dataset)} examples")
        
        # Show sample data
        loader = MaskedCritiqueDataLoader("data/sanitized_data_v4.csv")
        samples = loader.get_sample_data(3)
        
        print("\n📝 Sample data:")
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Prompt: {sample.get('prompt', 'N/A')[:100]}...")
            print(f"Critique: {sample.get('critique', 'N/A')[:100]}...")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure the data file exists and is accessible")
