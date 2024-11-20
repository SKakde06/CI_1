import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Disable matplotlib pyplot plotting to avoid conflicts with Streamlit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data():
    """Load and preprocess the customer data"""
    # Since we can't use the file path from Colab, we'll use a sample dataset
    data = pd.DataFrame({
        'region': ['North', 'South', 'East', 'West', 'Central'] * 20,
        'age': np.random.randint(18, 65, 100),
        'income': np.random.randint(30000, 150000, 100),
        'spending_score': np.random.randint(1, 100, 100),
        'purchase_amount': np.random.randint(100, 500, 100)
    })
    
    return data

def preprocess_data(data):
    """Preprocess the data for model training"""
    # Create high spender target variable
    data['high_spender'] = (data['purchase_amount'] > 300).astype(int)

    # Encode categorical features
    label_encoder = LabelEncoder()
    data['region'] = label_encoder.fit_transform(data['region'])

    # Separate features and target
    X = data.drop(columns=['high_spender', 'purchase_amount'])
    y = data['high_spender']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X_scaled, y):
    """Train a Random Forest Classifier"""
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """Calculate and return model evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    return metrics

def plot_confusion_matrix(y_test, y_pred):
    """Create a confusion matrix plot"""
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)
    
    # Add labels to the plot
    classes = ['Not High Spender', 'High Spender']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return fig

def main():
    st.title('Customer Purchasing Behavior Analysis')

    # Load and preprocess data
    data = load_data()
    X_scaled, y, scaler = preprocess_data(data)

    # Train model
    model, X_test, y_test, y_pred = train_model(X_scaled, y)

    # Evaluate model
    metrics = evaluate_model(y_test, y_pred)

    # Display metrics
    st.header('Model Performance Metrics')
    for metric, value in metrics.items():
        st.metric(metric, f'{value:.2f}')

    # Classification Report
    st.header('Classification Report')
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Confusion Matrix Plot
    st.header('Confusion Matrix')
    confusion_matrix_fig = plot_confusion_matrix(y_test, y_pred)
    st.pyplot(confusion_matrix_fig)

    # Optional: Feature Importance (if using Random Forest)
    st.header('Feature Importance')
    feature_importance = pd.DataFrame({
        'feature': ['region', 'age', 'income', 'spending_score'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    st.dataframe(feature_importance)

if __name__ == '__main__':
    main()
