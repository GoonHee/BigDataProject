# app.py
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Financial Fraud Detection Dashboard")
st.markdown("Big Data Analytics System using Machine Learning")

# Sidebar
st.sidebar.header("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

# Load data function
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        return None
    return df

# Train model function
@st.cache_resource
def train_models(df):
    # Features and target
    if 'Class' not in df.columns:
        st.error("Dataset must contain a 'Class' column (0=Normal, 1=Fraud)")
        return None, None
    
    X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
    y = df['Class']
    
    # Downsampling to balance the dataset
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0].sample(len(fraud), random_state=42)
    balanced_df = pd.concat([fraud, normal])
    
    X_bal = balanced_df.drop(['Class', 'Time'], axis=1, errors='ignore')
    y_bal = balanced_df['Class']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )
    
    # Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    return rf_model, (X_test, y_test, X_train, y_train)

# Generate demo data
def get_demo_data():
    """Generate synthetic demo data for testing"""
    np.random.seed(42)
    n_normal = 10000
    n_fraud = 50
    
    feature_cols = [f'V{i}' for i in range(1, 29)]
    
    normal_data = np.random.randn(n_normal, 28) * 0.5
    fraud_data = np.random.randn(n_fraud, 28) * 1.5 + 1
    
    X = np.vstack([normal_data, fraud_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
    
    df_demo = pd.DataFrame(X, columns=feature_cols)
    df_demo['Class'] = y
    df_demo['Amount'] = np.random.exponential(100, n_normal + n_fraud)
    df_demo['Time'] = np.arange(n_normal + n_fraud)
    
    return df_demo

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview", 
    "📈 Fraud Analysis", 
    "🤖 Model Performance", 
    "⚡ Real-time Prediction"
])

# Initialize variables
df = None
model = None
test_data = None

# Data loading section
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.sidebar.info("📌 No file uploaded. Use demo data below.")
    if st.sidebar.button("Use Demo Data"):
        df = get_demo_data()
        st.sidebar.success("Demo data loaded successfully")

if df is not None:
    with st.spinner("Training model..."):
        result = train_models(df)
        if result is not None and result[0] is not None:
            model, (X_test, y_test, X_train, y_train) = result
            test_data = (X_test, y_test)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            st.error("Model training failed")
else:
    st.info("👈 Please upload creditcard.csv or click 'Use Demo Data'")

# ==================== Tab 1: Data Overview ====================
with tab1:
    if df is not None:
        st.subheader("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            fraud_count = df['Class'].sum()
            st.metric("Fraud Transactions", fraud_count)
        with col3:
            normal_count = len(df) - fraud_count
            st.metric("Normal Transactions", normal_count)
        with col4:
            fraud_rate = fraud_count / len(df) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.4f}%")
        
        st.subheader("Data Preview (First 100 rows)")
        st.dataframe(df.head(100))
        
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())
    else:
        st.info("Waiting for data to load...")

# ==================== Tab 2: Fraud Analysis ====================
with tab2:
    if df is not None:
        st.subheader("📊 Fraud Transaction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution pie chart
            fig1 = px.pie(
                values=[len(df)-df['Class'].sum(), df['Class'].sum()],
                names=['Normal', 'Fraud'],
                title='Transaction Class Distribution',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Transaction amount distribution
            fig2 = px.box(
                df, x='Class', y='Amount',
                title='Transaction Amount Distribution (0=Normal, 1=Fraud)',
                labels={'Class': 'Transaction Type', 'Amount': 'Amount'},
                color='Class'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Time-based analysis (if Time column exists)
        if 'Time' in df.columns:
            st.subheader("Transaction Time Pattern")
            fig3 = px.histogram(
                df, x='Time', color='Class',
                title='Transaction Distribution over Time',
                nbins=50,
                labels={'Time': 'Time', 'count': 'Number of Transactions'}
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_cols = [c for c in numeric_cols if c.startswith('V')][:10] + ['Amount', 'Class']
        corr_matrix = df[corr_cols].corr()
        
        fig4, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig4)
    else:
        st.info("Waiting for data to load...")

# ==================== Tab 3: Model Performance ====================
with tab3:
    if model is not None and test_data is not None:
        st.subheader("🤖 Random Forest Model Performance")
        
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        with col3:
            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        with col4:
            st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fraud'], 
                    yticklabels=['Normal', 'Fraud'],
                    ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)
        
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic Curve')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        
        # Classification Report
        st.subheader("Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Feature Importance
        st.subheader("Feature Importance (Top 15)")
        feature_names = [c for c in df.columns if c not in ['Class', 'Time']][:model.feature_importances_.shape[0]]
        importances = model.feature_importances_[:len(feature_names)]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        ax_imp.barh(importance_df['Feature'], importance_df['Importance'])
        ax_imp.set_xlabel('Importance')
        ax_imp.set_title('Feature Importance Ranking')
        ax_imp.invert_yaxis()
        st.pyplot(fig_imp)
        
    else:
        st.info("Please load data and train the model first")

# ==================== Tab 4: Real-time Prediction ====================
with tab4:
    if model is not None:
        st.subheader("⚡ Real-time Fraud Prediction")
        st.markdown("Enter transaction features to predict fraud risk")
        
        # Get feature names
        feature_names = [c for c in df.columns if c not in ['Class', 'Time']]
        display_features = feature_names[:10]
        
        col1, col2 = st.columns(2)
        
        input_values = {}
        with col1:
            for f in display_features[:5]:
                input_values[f] = st.number_input(f"{f}", value=0.0, format="%.4f", key=f"feat_{f}")
        with col2:
            for f in display_features[5:10]:
                input_values[f] = st.number_input(f"{f}", value=0.0, format="%.4f", key=f"feat_{f}")
        
        # Amount field
        amount = st.number_input("Amount (Transaction Amount)", value=100.0, min_value=0.0)
        input_values['Amount'] = amount
        
        # Fill remaining features with 0
        for f in feature_names[10:]:
            input_values[f] = 0.0
        
        if st.button("🔍 Predict Fraud Risk", type="primary"):
            input_df = pd.DataFrame([input_values])[feature_names]
            proba = model.predict_proba(input_df)[0]
            prediction = model.predict(input_df)[0]
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("🚨 **HIGH RISK: Suspected Fraudulent Transaction**")
                else:
                    st.success("✅ **LOW RISK: Normal Transaction**")
            
            with col2:
                st.metric("Fraud Probability", f"{proba[1]*100:.2f}%")
            
            # Risk gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1]*100,
                title={'text': "Fraud Risk Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("Please load data and train the model first")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🔍 Fraud Detection System | Random Forest Classifier | Big Data Analytics Project
    </div>
    """,
    unsafe_allow_html=True
)