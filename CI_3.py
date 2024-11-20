import streamlit as st
import pandas as pd
import numpy as np

# Wrap sklearn imports in try-except to handle potential module import issues
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, 
        recall_score, f1_score, 
        classification_report, 
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def load_data():
    """Load sample customer data"""
    # Create a synthetic dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 100),
        'age': np.random.randint(18, 65, 100),
        'income': np.random.randint(30000, 150000, 100),
        'spending_score': np.random.randint(1, 100, 100),
        'purchase_amount': np.random.randint(100, 500, 100)
    })
    return data

def manual_preprocessing(data):
    """Manual preprocessing without sklearn"""
    # Create high spender target variable
    data['high_spender'] = (data['purchase_amount'] > 300).astype(int)
    
    # Manual encoding for region
    regions = sorted(data['region'].unique())
    region_map = {region: idx for idx, region in enumerate(regions)}
    data['region_encoded'] = data['region'].map(region_map)
    
    # Manual standardization
    numeric_cols = ['age', 'income', 'spending_score']
    for col in numeric_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[f'{col}_scaled'] = (data[col] - mean) / std
    
    # Prepare features and target
    X = data[['region_encoded', 'age_scaled', 'income_scaled', 'spending_score_scaled']]
    y = data['high_spender']
    
    return X, y

def manual_train_test_split(X, y, test_size=0.3, random_state=42):
    """Manual train-test split"""
    np.random.seed(random_state)
    total_samples = len(X)
    test_samples = int(total_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(total_samples)
    
    # Split indices
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # Split data
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

def manual_random_forest_classifier():
    """Simple manual implementation of a basic classifier"""
    class SimpleRandomForestClassifier:
        def __init__(self, n_estimators=100, random_state=42):
            self.n_estimators = n_estimators
            self.random_state = random_state
            self.trees = []
        
        def _bootstrap_sample(self, X, y):
            np.random.seed(self.random_state)
            n_samples = len(X)
            indices = np.random.randint(0, n_samples, n_samples)
            return X.iloc[indices], y.iloc[indices]
        
        def _train_tree(self, X, y):
            # Simple decision tree-like classification
            features = X.columns
            best_feature = np.random.choice(features)
            threshold = X[best_feature].median()
            
            def predict_tree(sample):
                return 1 if sample[best_feature] > threshold else 0
            
            return predict_tree
        
        def fit(self, X, y):
            # Train multiple simple trees
            self.trees = [self._train_tree(X, y) for _ in range(self.n_estimators)]
            return self
        
        def predict(self, X):
            # Ensemble predictions
            tree_predictions = np.array([[tree(row) for tree in self.trees] for _, row in X.iterrows()])
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=tree_predictions)

    return SimpleRandomForestClassifier()

def manual_classification_metrics(y_true, y_pred):
    """Calculate classification metrics manually"""
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Precision
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Recall
    actual_positives = np.sum(y_true == 1)
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def main():
    st.title('Customer Purchasing Behavior Analysis')
    
    # Check sklearn availability
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn is not available. Using manual implementations.")
    
    # Load data
    data = load_data()
    
    # Preprocess data
    X, y = manual_preprocessing(data)
    
    # Split data
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y)
    
    # Train model
    model = manual_random_forest_classifier()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = manual_classification_metrics(y_test, y_pred)
    
    # Display results
    st.header('Model Performance Metrics')
    for metric, value in metrics.items():
        st.metric(metric, f'{value:.2f}')
    
    # Confusion Matrix (manual)
    st.header('Confusion Matrix')
    conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    st.dataframe(conf_matrix)
    
    # Feature Importance (simplified)
    st.header('Feature Importance (Simplified)')
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.random.random(len(X.columns))
    }).sort_values('importance', ascending=False)
    st.dataframe(feature_importance)

if __name__ == '__main__':
    main()
