import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Financial Fraud Detection Dashboard")
st.markdown("Big Data Analytics System using Machine Learning")

# ==================== DATA PATH ====================
DATA_PATH = "C:/Users/ruxin/Documents/ruxin/Big Data 5011/Data/Bank_Transaction_Fraud_Detection.csv"

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ==================== PREPROCESS DATA ====================
@st.cache_data
def preprocess_data(df):
    """Clean and preprocess data for training"""
    target_col = 'Is_Fraud' if 'Is_Fraud' in df.columns else 'is_fraud'
    
    # Drop unnecessary columns
    cols_to_drop = [
        'Customer_ID', 'Customer_Name', 'Transaction_ID', 'Merchant_ID',
        'Customer_Contact', 'Customer_Email', 'Transaction_Description',
        'Transaction_Currency'
    ]
    df_clean = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Handle datetime
    if 'Transaction_Date' in df_clean.columns:
        df_clean['Transaction_Date'] = pd.to_datetime(df_clean['Transaction_Date'], format='%d-%m-%Y')
        df_clean['Day'] = df_clean['Transaction_Date'].dt.day
        df_clean['Month'] = df_clean['Transaction_Date'].dt.month
        df_clean['Day_of_Week'] = df_clean['Transaction_Date'].dt.dayofweek
        df_clean['Is_Weekend'] = (df_clean['Day_of_Week'] >= 5).astype(int)
        df_clean = df_clean.drop(columns=['Transaction_Date'])
    
    if 'Transaction_Time' in df_clean.columns:
        df_clean['Hour'] = pd.to_datetime(df_clean['Transaction_Time'], format='%H:%M:%S').dt.hour
        df_clean['Is_Night'] = ((df_clean['Hour'] >= 22) | (df_clean['Hour'] <= 5)).astype(int)
        df_clean = df_clean.drop(columns=['Transaction_Time'])
    
    # Encode categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col != target_col:
            df_clean[col] = df_clean[col].astype('category').cat.codes
    
    return df_clean

# ==================== TRAIN BOTH MODELS ====================
@st.cache_resource
def train_models(df):
    """Train both models: Random Forest and Logistic Regression"""
    
    target_col = 'Is_Fraud' if 'Is_Fraud' in df.columns else 'is_fraud'
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Balance dataset
    fraud = df[df[target_col] == 1]
    normal = df[df[target_col] == 0].sample(n=min(len(fraud) * 10, len(df[df[target_col] == 0])), random_state=42)
    balanced_df = pd.concat([fraud, normal])
    
    X_bal = balanced_df.drop(columns=[target_col])
    y_bal = balanced_df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )
    
    models = {}
    results = {}
    
    # 1. Random Forest
    with st.spinner("Training Random Forest..."):
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        results['Random Forest'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': auc(*roc_curve(y_test, y_proba)[:2]),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }
    
    # 2. Logistic Regression
    with st.spinner("Training Logistic Regression..."):
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
        y_pred = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)[:, 1]
        results['Logistic Regression'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': auc(*roc_curve(y_test, y_proba)[:2]),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }
    
    return models, results, X_train.columns, X_train

# ==================== PREPROCESS AND TRAIN ====================
df_clean = preprocess_data(df)
models, results, feature_names, X_train = train_models(df_clean)

# ==================== FIND BEST MODEL ====================
comparison_data = []
for name, metrics in results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'ROC AUC': metrics['roc_auc'] if metrics['roc_auc'] is not None else 0.5
    })
comparison_df = pd.DataFrame(comparison_data)
best_model_name = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs([
    "📊 Data Overview & EDA",
    "🤖 Model Performance Comparison",
    "⚡ Real-time Prediction"
])

# ==================== TAB 1: DATA OVERVIEW & EDA ====================
with tab1:
    st.header("📊 Data Overview & Exploratory Data Analysis")
    
    target_col = 'Is_Fraud' if 'Is_Fraud' in df.columns else 'is_fraud'
    fraud_count = df[target_col].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraud Transactions", int(fraud_count))
    with col3:
        st.metric("Normal Transactions", len(df) - int(fraud_count))
    with col4:
        st.metric("Fraud Rate", f"{fraud_count/len(df)*100:.2f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        fig1 = px.pie(
            values=[len(df) - fraud_count, fraud_count],
            names=['Normal', 'Fraud'],
            title='Transaction Class Distribution',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Transaction Amount Distribution")
        fig2 = px.box(
            df, x=target_col, y='Transaction_Amount',
            title='Amount Distribution by Class',
            color=target_col
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    if 'Hour' in df.columns:
        st.subheader("Transaction Hour Pattern")
        fig3 = px.histogram(
            df, x='Hour', color=target_col,
            title='Transactions by Hour',
            nbins=24,
            barmode='group'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Transaction_Type' in df.columns:
            st.subheader("Fraud Rate by Transaction Type")
            fraud_by_type = df.groupby('Transaction_Type')[target_col].mean().sort_values(ascending=False)
            fig4 = px.bar(
                x=fraud_by_type.index, y=fraud_by_type.values,
                title='Fraud Rate by Transaction Type',
                labels={'x': 'Transaction Type', 'y': 'Fraud Rate'}
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        if 'Account_Type' in df.columns:
            st.subheader("Fraud Rate by Account Type")
            fraud_by_account = df.groupby('Account_Type')[target_col].mean().sort_values(ascending=False)
            fig5 = px.bar(
                x=fraud_by_account.index, y=fraud_by_account.values,
                title='Fraud Rate by Account Type',
                labels={'x': 'Account Type', 'y': 'Fraud Rate'}
            )
            st.plotly_chart(fig5, use_container_width=True)

# ==================== TAB 2: MODEL PERFORMANCE ====================
with tab2:
    st.header("🤖 Model Performance Comparison")
    
    st.subheader("📊 Model Comparison Table")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='#90EE90'))
    
    st.subheader("📊 Model Performance Visualization")
    melted_df = comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    fig6 = px.bar(
        melted_df, x='Model', y='Score', color='Metric',
        title='Model Performance Comparison',
        barmode='group',
        text_auto='.3f'
    )
    fig6.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig6, use_container_width=True)
    
    st.subheader("📊 Confusion Matrices")
    col1, col2 = st.columns(2)
    
    for idx, (name, metrics) in enumerate(results.items()):
        with [col1, col2][idx]:
            st.markdown(f"**{name}**")
            cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'],
                       ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
    
    st.subheader("📊 ROC Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    
    for name, metrics in results.items():
        if metrics['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_proba'])
            ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {metrics["roc_auc"]:.4f})')
    
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
    
    # Feature Importance (Random Forest)
    if 'Random Forest' in models:
        st.subheader("📊 Feature Importance (Random Forest)")
        rf_model = models['Random Forest']
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(rf_model.feature_importances_)],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        ax_imp.barh(importance_df['Feature'], importance_df['Importance'])
        ax_imp.set_xlabel('Importance')
        ax_imp.set_title('Top 15 Feature Importance (Random Forest)')
        ax_imp.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_imp)
    
    st.success(f"🏆 **Best Model: {best_model_name}** (based on F1 Score)")

# ==================== TAB 3: REAL-TIME PREDICTION ====================
with tab3:
    st.header("⚡ Real-time Fraud Prediction")
    st.info(f"**Using Best Model: {best_model_name}**")
    
    model = models[best_model_name]
    
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    
    with col1:
        # Gender: Female/Male -> 0/1
        gender_input = st.selectbox("Gender", ["Female", "Male"])
        input_values['Gender'] = 1 if gender_input == "Male" else 0
        
        age = st.slider("Age", 18, 100, 35)
        input_values['Age'] = age
        
        if 'Account_Type' in feature_names:
            account_type = st.selectbox("Account Type", ["Checking", "Savings", "Business"])
            account_map = {'Checking': 0, 'Savings': 1, 'Business': 2}
            input_values['Account_Type'] = account_map.get(account_type, 0)
        
        transaction_amount = st.number_input("Transaction Amount", min_value=0.01, value=500.0)
        input_values['Transaction_Amount'] = transaction_amount
    
    with col2:
        if 'Transaction_Type' in feature_names:
            trans_type = st.selectbox("Transaction Type", ["Credit", "Debit", "Transfer", "Withdrawal", "Bill Payment"])
            trans_map = {'Credit': 0, 'Debit': 1, 'Transfer': 2, 'Withdrawal': 3, 'Bill Payment': 4}
            input_values['Transaction_Type'] = trans_map.get(trans_type, 0)
        
        if 'Merchant_Category' in feature_names:
            merchant = st.selectbox("Merchant Category", ["Retail", "Restaurant", "Groceries", "Entertainment", "Travel", "Utilities", "Other"])
            merchant_map = {'Retail': 0, 'Restaurant': 1, 'Groceries': 2, 'Entertainment': 3, 'Travel': 4, 'Utilities': 5, 'Other': 6}
            input_values['Merchant_Category'] = merchant_map.get(merchant, 0)
        
        account_balance = st.number_input("Account Balance", min_value=0.0, value=5000.0)
        input_values['Account_Balance'] = account_balance
        
        if 'Transaction_Device' in feature_names:
            device = st.selectbox("Transaction Device", ["Mobile", "Desktop", "Tablet", "ATM", "POS"])
            device_map = {'Mobile': 0, 'Desktop': 1, 'Tablet': 2, 'ATM': 3, 'POS': 4}
            input_values['Transaction_Device'] = device_map.get(device, 0)
    
    with col3:
        hour = st.slider("Hour", 0, 23, 14)
        input_values['Hour'] = hour
        
        day = st.slider("Day", 1, 31, 15)
        input_values['Day'] = day
        
        month = st.slider("Month", 1, 12, 6)
        input_values['Month'] = month
        
        # Auto-calculated features
        input_values['Is_Night'] = 1 if (hour >= 22 or hour <= 5) else 0
        input_values['Is_Weekend'] = 1 if (day % 7 == 0) else 0
    
    # Fill remaining features with 0
    for f in feature_names:
        if f not in input_values:
            input_values[f] = 0.0
    
    if st.button("🔍 Predict Fraud Risk", type="primary"):
        input_df = pd.DataFrame([input_values])[feature_names]
        
        # Predict
        y_proba = model.predict_proba(input_df)[0][1]
        y_pred = model.predict(input_df)[0]
        
        try:
            from src.database import insert_transaction
            
            transaction_data = (
                gender_input,           
                age,                    
                account_type,           
                transaction_amount,     
                trans_type,             
                merchant,               
                account_balance,        
                device,                 
                device,                 
                'USD',                  
                month,                  
                day,                    
                hour,                   
                None,                   
                float(y_proba),         
                int(y_pred),            
                best_model_name        
            )
            
            transaction_id = insert_transaction(transaction_data)
                
        except Exception as e:
            st.warning(f"⚠️ Database save skipped: {e}")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if y_pred == 1:
                st.error("🚨 **HIGH RISK: Fraudulent Transaction Detected!**")
            else:
                st.success("✅ **LOW RISK: Transaction appears normal**")
        
        with col2:
            st.metric("Fraud Probability", f"{y_proba*100:.2f}%")
        
        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_proba * 100,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Transaction Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Gender:** {gender_input}")
            st.write(f"**Age:** {age}")
            st.write(f"**Account Type:** {account_type if 'account_type' in locals() else 'N/A'}")
        with col2:
            st.write(f"**Transaction Type:** {trans_type if 'trans_type' in locals() else 'N/A'}")
            st.write(f"**Merchant Category:** {merchant if 'merchant' in locals() else 'N/A'}")
            st.write(f"**Transaction Amount:** ${transaction_amount:,.2f}")
        with col3:
            st.write(f"**Hour:** {hour}:00")
            st.write(f"**Account Balance:** ${account_balance:,.2f}")
            st.write(f"**Model Used:** {best_model_name}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🔍 Fraud Detection System | Random Forest vs Logistic Regression | Big Data Analytics Project
    </div>
    """,
    unsafe_allow_html=True
)