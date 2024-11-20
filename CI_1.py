import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

class OptimizedMLPipeline:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        # Efficient data preprocessing using vectorized operations
        df = data.copy()
        
        # Create target variable using vectorized operation
        df['high_spender'] = (df['purchase_amount'] > 300).astype(int)
        
        # Drop columns efficiently
        df.drop(columns=['user_id', 'purchase_amount'], inplace=True)
        
        # Encode categorical features using pre-initialized encoder
        df['region'] = self.label_encoder.fit_transform(df['region'])
        
        return df
    
    def create_train_test_split(self, df):
        # Efficient feature-target split
        X = df.drop(columns=['high_spender'])
        y = df['high_spender']
        
        # Use smaller chunk size for train_test_split
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def train_model(self, X_train, y_train):
        # Create an efficient pipeline
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,  # Limit tree depth
                min_samples_split=10,  # Increase min samples for splitting
                n_jobs=self.n_jobs,
                random_state=42
            ))
        ])
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def evaluate_model(self, pipeline, X_test, y_test):
        # Make predictions in batches
        batch_size = 1000
        predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test.iloc[i:i+batch_size]
            batch_pred = pipeline.predict(batch)
            predictions.extend(batch_pred)
        
        y_pred = np.array(predictions)
        
        # Calculate metrics efficiently
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics, y_pred

def run_optimized_pipeline(file_path):
    # Read data efficiently by specifying dtypes
    dtypes = {
        'user_id': 'int32',
        'annual_income': 'float32',
        'purchase_amount': 'float32'
    }
    
    data = pd.read_csv(file_path, dtype=dtypes)
    
    # Initialize and run pipeline
    pipeline = OptimizedMLPipeline()
    
    # Process data
    df = pipeline.preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = pipeline.create_train_test_split(df)
    
    # Train model
    model_pipeline = pipeline.train_model(X_train, y_train)
    
    # Evaluate model
    metrics, predictions = pipeline.evaluate_model(model_pipeline, X_test, y_test)
    
    return metrics, model_pipeline

# Example usage
if __name__ == "__main__":
    file_path = 'Customer Purchasing Behaviors.csv'
    metrics, model = run_optimized_pipeline(file_path)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")
