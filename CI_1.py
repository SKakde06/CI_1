import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="ML Pipeline", layout="wide")

class OptimizedMLPipeline:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        df = data.copy()
        df['high_spender'] = (df['purchase_amount'] > 300).astype(int)
        df.drop(columns=['user_id', 'purchase_amount'], inplace=True)
        df['region'] = self.label_encoder.fit_transform(df['region'])
        return df
    
    def create_train_test_split(self, df):
        X = df.drop(columns=['high_spender'])
        y = df['high_spender']
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def train_model(self, X_train, y_train):
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                n_jobs=self.n_jobs,
                random_state=42
            ))
        ])
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def evaluate_model(self, pipeline, X_test, y_test):
        batch_size = 1000
        predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test.iloc[i:i+batch_size]
            batch_pred = pipeline.predict(batch)
            predictions.extend(batch_pred)
        
        y_pred = np.array(predictions)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics, y_pred

def main():
    st.title("Machine Learning Pipeline")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read data
            dtypes = {
                'user_id': 'int32',
                'annual_income': 'float32',
                'purchase_amount': 'float32'
            }
            data = pd.read_csv(uploaded_file, dtype=dtypes)
            
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Initialize and run pipeline
            pipeline = OptimizedMLPipeline()
            
            with st.spinner("Processing data..."):
                df = pipeline.preprocess_data(data)
            
            with st.spinner("Splitting data..."):
                X_train, X_test, y_train, y_test = pipeline.create_train_test_split(df)
            
            with st.spinner("Training model..."):
                model_pipeline = pipeline.train_model(X_train, y_train)
            
            with st.spinner("Evaluating model..."):
                metrics, predictions = pipeline.evaluate_model(model_pipeline, X_test, y_test)
            
            # Display results
            st.subheader("Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.2f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.2f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1']:.2f}")
            
            # Plot feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model_pipeline.named_steps['classifier'].feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title('Feature Importance')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
