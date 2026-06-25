import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestFraudDetector:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.is_trained = False
        self.feature_names = None
        self.metrics = {}
    
    def train(self, X, y, feature_names=None):
        self.feature_names = feature_names
        
        if y.sum() < 2:
            logger.error("Not enough fraud samples to train")
            return {'error': 'Insufficient fraud samples'}
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Fraud ratio in training: {y_train.mean():.4f}")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5
        }
        
        self.is_trained = True
        logger.info(f"✅ Random Forest trained! F1 Score: {self.metrics['f1_score']:.4f}")
        return self.metrics
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, top_n=15):
        if not self.is_trained or self.feature_names is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(self.model.feature_importances_)],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


class LogisticRegressionDetector:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        self.is_trained = False
        self.metrics = {}
    
    def train(self, X, y, feature_names=None):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.5
        }
        
        self.is_trained = True
        logger.info(f"✅ Logistic Regression trained! F1: {self.metrics['f1_score']:.4f}")
        return self.metrics
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def compare_models(X, y, feature_names):
    """Train and compare two models: Random Forest and Logistic Regression"""
    print("\n" + "=" * 60)
    print("🤖 Training and Comparing Fraud Detection Models")
    print("=" * 60)
    
    print(f"\n📊 Using {X.shape[0]:,} samples with {X.shape[1]} features")
    print(f"📊 Fraud ratio: {y.mean():.4f} ({int(y.sum()):,} fraud cases)")
    
    models = {
        'Random Forest': RandomForestFraudDetector(),
        'Logistic Regression': LogisticRegressionDetector()
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        
        try:
            metrics = model.train(X, y, feature_names)
            results[name] = metrics
            trained_models[name] = model
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results).T
    
    available_cols = [col for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] 
                     if col in comparison_df.columns]
    
    comparison_df = comparison_df[available_cols]
    
    column_rename = {
        'accuracy': 'Accuracy',
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'roc_auc': 'ROC AUC'
    }
    comparison_df = comparison_df.rename(columns=column_rename)
    
    print("\n" + "=" * 60)
    print("📊 Model Comparison Results")
    print("=" * 60)
    print(comparison_df.round(4))
    
    return comparison_df, trained_models


def select_best_model(comparison_df, metric='F1 Score'):
    """Select the best model based on metric"""
    if metric in comparison_df.columns:
        best_model_name = comparison_df[metric].idxmax()
    elif 'F1 Score' in comparison_df.columns:
        best_model_name = comparison_df['F1 Score'].idxmax()
    elif 'Accuracy' in comparison_df.columns:
        best_model_name = comparison_df['Accuracy'].idxmax()
    else:
        best_model_name = comparison_df.iloc[:, 0].idxmax()
    
    return best_model_name


def load_and_prepare_data(data_path, target_col='Is_Fraud'):
    """Load preprocessed data and prepare for model training"""
    print(f"\n📂 Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} records")
    print(f"📋 Columns: {list(df.columns)}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found! Available: {list(df.columns)}")
    
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📊 Fraud cases: {int(y.sum()):,} ({y.mean():.4f})")
    
    return X, y, feature_cols

# ==================== MAIN ====================
if __name__ == "__main__":
    DATA_PATH = "C:/Users/ruxin/Documents/ruxin/Big Data 5011/Data/preprocessed_data.csv"
    
    print("=" * 60)
    print("🔍 FRAUD DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    X, y, feature_names = load_and_prepare_data(DATA_PATH, target_col='Is_Fraud')
    
    comparison, models = compare_models(X, y, feature_names)
    
    best = select_best_model(comparison, 'F1 Score')
    print(f"\n🏆 Best model: {best}")
    
    best_model = models[best]
    print(f"\n📈 Best Model Performance:")
    for metric, value in best_model.metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")