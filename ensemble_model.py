"""
Ensemble Model Module for Trading Bot

Implements stacking and voting ensemble methods to improve prediction accuracy
by combining multiple base models (XGBoost, Random Forest, Gradient Boosting).
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Stacking ensemble combining multiple base models with a meta-learner.
    
    Base Models (Level 0):
    - XGBoost Classifier
    - Random Forest Classifier
    - Gradient Boosting Classifier
    
    Meta-Learner (Level 1):
    - Logistic Regression (combines base predictions)
    """
    
    def __init__(self, use_cv=True):
        """
        Initialize stacking ensemble.
        
        Args:
            use_cv: If True, use cross-validated predictions for meta-learner
        """
        self.use_cv = use_cv
        self.base_models = {}
        self.meta_model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the stacking ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("🔧 Training Stacking Ensemble...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        
        # 1. Train base models
        logger.info("Training base models...")
        
        # XGBoost
        logger.info("  - XGBoost Classifier...")
        self.base_models['xgboost'] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )
        self.base_models['xgboost'].fit(X_train, y_train)
        
        # Random Forest
        logger.info("  - Random Forest Classifier...")
        self.base_models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        self.base_models['random_forest'].fit(X_train, y_train)
        
        # Gradient Boosting
        logger.info("  - Gradient Boosting Classifier...")
        self.base_models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        self.base_models['gradient_boosting'].fit(X_train, y_train)
        
        # 2. Create meta-features
        logger.info("Creating meta-features...")
        
        if self.use_cv:
            # Use cross-validated predictions (better generalization)
            meta_X_train = np.column_stack([
                cross_val_predict(self.base_models['xgboost'], X_train, y_train, 
                                cv=5, method='predict_proba', n_jobs=-1)[:, 1],
                cross_val_predict(self.base_models['random_forest'], X_train, y_train, 
                                cv=5, method='predict_proba', n_jobs=-1)[:, 1],
                cross_val_predict(self.base_models['gradient_boosting'], X_train, y_train, 
                                cv=5, method='predict_proba', n_jobs=-1)[:, 1]
            ])
        else:
            # Use direct predictions
            meta_X_train = np.column_stack([
                self.base_models['xgboost'].predict_proba(X_train)[:, 1],
                self.base_models['random_forest'].predict_proba(X_train)[:, 1],
                self.base_models['gradient_boosting'].predict_proba(X_train)[:, 1]
            ])
        
        # 3. Train meta-learner
        logger.info("Training meta-learner (Logistic Regression)...")
        self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_model.fit(meta_X_train, y_train)
        
        # 4. Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_acc = (val_pred == y_val).mean()
            logger.info(f"✅ Ensemble Validation Accuracy: {val_acc:.4f}")
        
        logger.info("✅ Stacking Ensemble training complete!")
        
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of probabilities for class 1
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get predictions from base models
        meta_X = np.column_stack([
            self.base_models['xgboost'].predict_proba(X)[:, 1],
            self.base_models['random_forest'].predict_proba(X)[:, 1],
            self.base_models['gradient_boosting'].predict_proba(X)[:, 1]
        ])
        
        # Meta-learner prediction
        return self.meta_model.predict_proba(meta_X)[:, 1]
    
    def predict(self, X, threshold=0.65):
        """
        Predict class labels.
        
        Args:
            X: Features
            threshold: Decision threshold
            
        Returns:
            Array of predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, path):
        """Save ensemble to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'use_cv': self.use_cv
        }, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path):
        """Load ensemble from disk."""
        data = joblib.load(path)
        self.base_models = data['base_models']
        self.meta_model = data['meta_model']
        self.feature_names = data.get('feature_names')
        self.use_cv = data.get('use_cv', True)
        logger.info(f"Ensemble loaded from {path}")


class VotingEnsemble:
    """
    Simple voting ensemble with weighted predictions.
    
    Combines predictions from multiple models using weighted voting.
    """
    
    def __init__(self, weights=None):
        """
        Initialize voting ensemble.
        
        Args:
            weights: Dict of model weights (default: equal weights)
        """
        self.weights = weights or {'xgboost': 0.4, 'random_forest': 0.3, 'gradient_boosting': 0.3}
        self.models = {}
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the voting ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("🗳️ Training Voting Ensemble...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        
        # Train models
        logger.info("Training models...")
        
        self.models['xgboost'] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )
        self.models['xgboost'].fit(X_train, y_train)
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.models['gradient_boosting'].fit(X_train, y_train)
        
        # Evaluate
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_acc = (val_pred == y_val).mean()
            logger.info(f"✅ Voting Ensemble Validation Accuracy: {val_acc:.4f}")
        
        logger.info("✅ Voting Ensemble training complete!")
        
    def predict_proba(self, X):
        """Predict class probabilities using weighted voting."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Weighted average of probabilities
        proba = np.zeros(len(X))
        for model_name, weight in self.weights.items():
            proba += weight * self.models[model_name].predict_proba(X)[:, 1]
        
        return proba
    
    def predict(self, X, threshold=0.65):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def save(self, path):
        """Save ensemble to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'models': self.models,
            'weights': self.weights,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Voting ensemble saved to {path}")
    
    def load(self, path):
        """Load ensemble from disk."""
        data = joblib.load(path)
        self.models = data['models']
        self.weights = data['weights']
        self.feature_names = data.get('feature_names')
        logger.info(f"Voting ensemble loaded from {path}")


def load_ensemble_model(ensemble_type='stacking'):
    """
    Load ensemble model from disk.
    
    Args:
        ensemble_type: 'stacking' or 'voting'
        
    Returns:
        Ensemble model or None if not found
    """
    path = f"models/ensemble_{ensemble_type}.pkl"
    
    if not os.path.exists(path):
        return None
    
    try:
        if ensemble_type == 'stacking':
            ensemble = StackingEnsemble()
        else:
            ensemble = VotingEnsemble()
        
        ensemble.load(path)
        return ensemble
    except Exception as e:
        logger.error(f"Failed to load ensemble: {e}")
        return None
