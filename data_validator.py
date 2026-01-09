"""
Data Validation Module for ML Training

Validates data quality before model training to prevent issues with:
- Insufficient data
- Time series gaps
- Outliers
- Class imbalance
- NaN values
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates training data quality."""
    
    @staticmethod
    def check_data_quality(df):
        """
        Validates data before training.
        
        Args:
            df: DataFrame with training data
            
        Returns:
            tuple: (is_valid, issues_list)
        """
        issues = []
        
        # 1. Check for sufficient data
        if len(df) < 1000:
            issues.append(f"Insufficient data: {len(df)} rows (need 1000+)")
        
        # 2. Check for gaps in time series
        if 'time' in df.columns:
            try:
                df_sorted = df.sort_values('time')
                time_diffs = df_sorted['time'].diff()
                
                # Get expected interval (mode of differences)
                if len(time_diffs) > 1:
                    expected_diff = time_diffs.mode()[0] if not time_diffs.mode().empty else pd.Timedelta(seconds=60)
                    
                    # Find gaps larger than 2x expected
                    gaps = time_diffs[time_diffs > expected_diff * 2]
                    if len(gaps) > 0:
                        issues.append(f"Found {len(gaps)} time gaps (largest: {gaps.max()})")
            except Exception as e:
                logger.warning(f"Could not check time gaps: {e}")
        
        # 3. Check for outliers (price spikes)
        for col in ['open', 'close', 'min', 'max']:
            if col in df.columns:
                try:
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    if std > 0:  # Avoid division by zero
                        z_scores = np.abs((df[col] - mean) / std)
                        outliers = z_scores > 5
                        if outliers.sum() > 0:
                            issues.append(f"{col}: {outliers.sum()} outliers detected (>5 sigma)")
                except Exception as e:
                    logger.warning(f"Could not check outliers for {col}: {e}")
        
        # 4. Check class balance
        if 'outcome' in df.columns:
            try:
                balance = df['outcome'].value_counts(normalize=True)
                if len(balance) > 0 and balance.min() < 0.3:  # Less than 30% minority class
                    issues.append(f"Class imbalance: {balance.to_dict()}")
            except Exception as e:
                logger.warning(f"Could not check class balance: {e}")
        
        # 5. Check for NaN after feature engineering
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            nan_counts = df[nan_cols].isna().sum()
            issues.append(f"NaN values in {len(nan_cols)} columns: {nan_counts.to_dict()}")
        
        # 6. Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✅ Data quality check passed")
        else:
            logger.warning(f"⚠️ Data quality issues found: {len(issues)}")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    @staticmethod
    def clean_data(df):
        """
        Cleans data by removing outliers and filling gaps.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame: Cleaned data
        """
        df = df.copy()
        initial_rows = len(df)
        
        # 1. Remove extreme outliers (5 sigma)
        for col in ['open', 'close', 'min', 'max']:
            if col in df.columns:
                try:
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    if std > 0:
                        df = df[(df[col] >= mean - 5*std) & (df[col] <= mean + 5*std)]
                except Exception as e:
                    logger.warning(f"Could not remove outliers from {col}: {e}")
        
        # 2. Remove duplicates
        df = df.drop_duplicates()
        
        # 3. Forward fill small gaps (max 3 candles)
        # Only for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill', limit=3)
        
        # 4. Drop remaining NaNs
        df = df.dropna()
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"🧹 Cleaned data: removed {rows_removed} rows ({rows_removed/initial_rows*100:.1f}%)")
        
        return df
    
    @staticmethod
    def validate_and_clean(df, strict=False):
        """
        Convenience method: validates and cleans data in one call.
        
        Args:
            df: DataFrame to validate and clean
            strict: If True, raises exception on validation failure
            
        Returns:
            DataFrame: Cleaned and validated data
            
        Raises:
            ValueError: If strict=True and validation fails
        """
        # First check quality
        is_valid, issues = DataValidator.check_data_quality(df)
        
        if not is_valid:
            if strict:
                raise ValueError(f"Data validation failed: {issues}")
            else:
                logger.warning("Data has quality issues. Attempting to clean...")
        
        # Clean the data
        df_clean = DataValidator.clean_data(df)
        
        # Re-validate after cleaning
        is_valid_after, issues_after = DataValidator.check_data_quality(df_clean)
        
        if not is_valid_after and strict:
            raise ValueError(f"Data still invalid after cleaning: {issues_after}")
        
        return df_clean


# Convenience functions for backward compatibility
def check_data_quality(df):
    """Validates data quality. Returns (is_valid, issues_list)."""
    return DataValidator.check_data_quality(df)


def clean_data(df):
    """Cleans data by removing outliers and filling gaps."""
    return DataValidator.clean_data(df)
