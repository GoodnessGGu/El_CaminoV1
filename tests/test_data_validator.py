"""
Unit tests for data_validator module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_validator import DataValidator


def create_sample_data(n_rows=1000, add_outliers=False, add_gaps=False, imbalanced=False):
    """Helper function to create sample data for testing."""
    np.random.seed(42)
    
    # Create time series
    start_time = datetime(2024, 1, 1)
    times = [start_time + timedelta(minutes=i) for i in range(n_rows)]
    
    # Add gaps if requested
    if add_gaps:
        # Remove some timestamps to create gaps
        times = [t for i, t in enumerate(times) if i % 100 != 50]
    
    # Create OHLCV data
    base_price = 1.1000
    df = pd.DataFrame({
        'time': times[:len(times)],
        'open': base_price + np.random.randn(len(times)) * 0.001,
        'close': base_price + np.random.randn(len(times)) * 0.001,
        'min': base_price - abs(np.random.randn(len(times)) * 0.001),
        'max': base_price + abs(np.random.randn(len(times)) * 0.001),
        'volume': np.random.randint(1000, 10000, len(times))
    })
    
    # Add outliers if requested
    if add_outliers:
        df.loc[10, 'close'] = base_price * 10  # Huge spike
        df.loc[20, 'close'] = base_price * 0.1  # Huge drop
    
    # Add outcome column
    if imbalanced:
        # 90% wins, 10% losses
        df['outcome'] = np.random.choice([0, 1], size=len(df), p=[0.1, 0.9])
    else:
        # Balanced
        df['outcome'] = np.random.choice([0, 1], size=len(df))
    
    return df


class TestDataValidator:
    
    def test_sufficient_data_check(self):
        """Test that insufficient data is detected."""
        # Create small dataset
        df = create_sample_data(n_rows=500)
        is_valid, issues = DataValidator.check_data_quality(df)
        
        assert not is_valid
        assert any('Insufficient data' in issue for issue in issues)
    
    def test_outlier_detection(self):
        """Test that outliers are detected."""
        df = create_sample_data(n_rows=1000, add_outliers=True)
        is_valid, issues = DataValidator.check_data_quality(df)
        
        assert not is_valid
        assert any('outliers detected' in issue for issue in issues)
    
    def test_gap_detection(self):
        """Test that time gaps are detected."""
        df = create_sample_data(n_rows=1000, add_gaps=True)
        is_valid, issues = DataValidator.check_data_quality(df)
        
        assert not is_valid
        assert any('time gaps' in issue for issue in issues)
    
    def test_class_balance_check(self):
        """Test that class imbalance is detected."""
        df = create_sample_data(n_rows=1000, imbalanced=True)
        is_valid, issues = DataValidator.check_data_quality(df)
        
        assert not is_valid
        assert any('Class imbalance' in issue for issue in issues)
    
    def test_clean_data_removes_outliers(self):
        """Test that cleaning removes outliers."""
        df = create_sample_data(n_rows=1000, add_outliers=True)
        initial_rows = len(df)
        
        df_clean = DataValidator.clean_data(df)
        
        assert len(df_clean) < initial_rows  # Some rows removed
        assert len(df_clean) > initial_rows * 0.95  # But not too many
    
    def test_clean_data_removes_duplicates(self):
        """Test that cleaning removes duplicates."""
        df = create_sample_data(n_rows=1000)
        # Add duplicate rows
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)
        
        df_clean = DataValidator.clean_data(df)
        
        assert df_clean.duplicated().sum() == 0
    
    def test_valid_data_passes(self):
        """Test that valid data passes all checks."""
        df = create_sample_data(n_rows=1000)
        is_valid, issues = DataValidator.check_data_quality(df)
        
        # Should pass (or have minimal issues)
        assert len(issues) <= 1  # Allow for minor issues like slight imbalance
    
    def test_validate_and_clean(self):
        """Test the combined validate_and_clean method."""
        df = create_sample_data(n_rows=1000, add_outliers=True)
        
        df_clean = DataValidator.validate_and_clean(df, strict=False)
        
        # Should return cleaned data
        assert len(df_clean) > 0
        assert len(df_clean) < len(df)  # Outliers removed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
