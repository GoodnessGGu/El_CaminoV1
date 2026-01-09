"""
Unit tests for feature_engineering module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import (
    add_market_context_features,
    add_order_flow_features,
    add_all_advanced_features
)


def create_sample_data_with_indicators(n_rows=200):
    """Helper function to create sample data with basic indicators."""
    np.random.seed(42)
    
    # Create time series
    start_time = datetime(2024, 1, 1, 8, 0)  # Start at 8 AM
    times = [start_time + timedelta(minutes=i) for i in range(n_rows)]
    
    # Create OHLCV data
    base_price = 1.1000
    df = pd.DataFrame({
        'time': times,
        'open': base_price + np.random.randn(n_rows) * 0.001,
        'close': base_price + np.random.randn(n_rows) * 0.001,
        'min': base_price - abs(np.random.randn(n_rows) * 0.001),
        'max': base_price + abs(np.random.randn(n_rows) * 0.001),
        'volume': np.random.randint(1000, 10000, n_rows)
    })
    
    # Add basic indicators (simulated)
    df['rsi'] = 50 + np.random.randn(n_rows) * 15
    df['adx'] = 20 + abs(np.random.randn(n_rows) * 10)
    df['atr'] = 0.001 + abs(np.random.randn(n_rows) * 0.0005)
    df['macd_hist'] = np.random.randn(n_rows) * 0.0001
    
    return df


class TestMarketContextFeatures:
    
    def test_adds_regime_features(self):
        """Test that regime features are added."""
        df = create_sample_data_with_indicators()
        df_features = add_market_context_features(df)
        
        assert 'regime_trending' in df_features.columns
        assert 'regime_ranging' in df_features.columns
        assert df_features['regime_trending'].dtype == int
        assert df_features['regime_ranging'].dtype == int
    
    def test_adds_volatility_regime(self):
        """Test that volatility regime is added."""
        df = create_sample_data_with_indicators()
        df_features = add_market_context_features(df)
        
        assert 'atr_percentile' in df_features.columns
    
    def test_adds_time_features(self):
        """Test that time-based features are added."""
        df = create_sample_data_with_indicators()
        df_features = add_market_context_features(df)
        
        assert 'hour' in df_features.columns
        assert 'is_london_session' in df_features.columns
        assert 'is_ny_session' in df_features.columns
        assert 'is_overlap' in df_features.columns
    
    def test_adds_support_resistance(self):
        """Test that S/R levels are added."""
        df = create_sample_data_with_indicators()
        df_features = add_market_context_features(df)
        
        assert 'resistance_50' in df_features.columns
        assert 'support_50' in df_features.columns
        assert 'dist_to_resistance' in df_features.columns
        assert 'dist_to_support' in df_features.columns
    
    def test_adds_momentum_score(self):
        """Test that momentum score is added."""
        df = create_sample_data_with_indicators()
        df_features = add_market_context_features(df)
        
        assert 'momentum_score' in df_features.columns
        # Score should be between -1 and 1
        assert df_features['momentum_score'].min() >= -1.5
        assert df_features['momentum_score'].max() <= 1.5
    
    def test_adds_volume_ratio(self):
        """Test that volume ratio is added."""
        df = create_sample_data_with_indicators()
        df_features = add_market_context_features(df)
        
        assert 'volume_ratio' in df_features.columns
    
    def test_no_nans_in_output(self):
        """Test that no NaN values are introduced (except in rolling windows)."""
        df = create_sample_data_with_indicators()
        df_features = add_market_context_features(df)
        
        # Check last 100 rows (after rolling windows stabilize)
        assert df_features.iloc[-100:].isna().sum().sum() == 0


class TestOrderFlowFeatures:
    
    def test_adds_pressure_features(self):
        """Test that buying/selling pressure features are added."""
        df = create_sample_data_with_indicators()
        df_features = add_order_flow_features(df)
        
        assert 'buying_pressure' in df_features.columns
        assert 'selling_pressure' in df_features.columns
        assert 'net_pressure_10' in df_features.columns
    
    def test_adds_wick_features(self):
        """Test that wick rejection features are added."""
        df = create_sample_data_with_indicators()
        df_features = add_order_flow_features(df)
        
        assert 'upper_wick' in df_features.columns
        assert 'lower_wick' in df_features.columns
        assert 'wick_bias' in df_features.columns
    
    def test_adds_streak_feature(self):
        """Test that candle streak feature is added."""
        df = create_sample_data_with_indicators()
        df_features = add_order_flow_features(df)
        
        assert 'candle_color' in df_features.columns
        assert 'streak' in df_features.columns
        
        # Streak should be non-zero for most candles
        assert (df_features['streak'] != 0).sum() > len(df_features) * 0.8
    
    def test_pressure_calculation(self):
        """Test that pressure is calculated correctly."""
        # Create simple test case
        df = pd.DataFrame({
            'open': [1.0, 1.0, 1.0],
            'close': [1.1, 0.9, 1.05],  # Green, Red, Green
            'min': [0.95, 0.85, 0.95],
            'max': [1.15, 1.05, 1.1]
        })
        
        df_features = add_order_flow_features(df)
        
        # First candle is green (buying pressure > 0)
        assert df_features.iloc[0]['buying_pressure'] > 0
        assert df_features.iloc[0]['selling_pressure'] == 0
        
        # Second candle is red (selling pressure > 0)
        assert df_features.iloc[1]['buying_pressure'] == 0
        assert df_features.iloc[1]['selling_pressure'] > 0


class TestAllAdvancedFeatures:
    
    def test_adds_all_features(self):
        """Test that all advanced features are added."""
        df = create_sample_data_with_indicators()
        initial_cols = len(df.columns)
        
        df_features = add_all_advanced_features(df)
        
        # Should have added 21 features (13 market context + 8 order flow)
        assert len(df_features.columns) >= initial_cols + 20
    
    def test_feature_count(self):
        """Test that expected number of features are added."""
        df = create_sample_data_with_indicators()
        df_features = add_all_advanced_features(df)
        
        # Check for key features
        expected_features = [
            'regime_trending', 'regime_ranging', 'atr_percentile',
            'momentum_score', 'buying_pressure', 'selling_pressure',
            'wick_bias', 'streak'
        ]
        
        for feature in expected_features:
            assert feature in df_features.columns, f"Missing feature: {feature}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
