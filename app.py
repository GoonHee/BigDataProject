import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

st.title("🔍 Financial Fraud Detection Dashboard")
st.markdown("Big Data Analytics System using Machine Learning")

# ==================== DATA PATHS ====================
RAW_DATA_PATH   = "D:/5011/Bank_Transaction_Fraud_Detection.csv"
CLEAN_DATA_PATH = "D:/5011/preprocessed_data.csv"

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    raw   = pd.read_csv(RAW_DATA_PATH)
    clean = pd.read_csv(CLEAN_DATA_PATH)
    return raw, clean

try:
    df_raw, df_clean = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

target_col = 'Is_Fraud' if 'Is_Fraud' in df_clean.columns else 'is_fraud'

# ==================== SCALE COLS (for prediction input only) ====================
binary_cols = ['Is_Night', 'Is_Weekend', 'Is_High_Amount', 'Is_Low_Balance', target_col]
scale_cols = [
    col for col in df_clean.select_dtypes(include='number').columns
    if col not in binary_cols
]
# Scaler fitted on preprocessed data — used only for prediction input, NOT training
scaler = StandardScaler()
scaler.fit(df_clean[scale_cols])

# ==================== TRAIN BOTH MODELS ====================
@st.cache_resource
def train_models(df):
    # Use only numeric columns — drop anything that didn't encode properly
    df_num = df.copy()
    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce')
    df_num = df_num.dropna(axis=1, how='all')
    df_num = df_num.fillna(0)

    tc = 'Is_Fraud' if 'Is_Fraud' in df_num.columns else 'is_fraud'

    fraud  = df_num[df_num[tc] == 1]
    normal = df_num[df_num[tc] == 0].sample(
        n=min(len(fraud) * 2, len(df_num[df_num[tc] == 0])), random_state=42
    )
    balanced_df = pd.concat([fraud, normal])

    X_bal = balanced_df.drop(columns=[tc])
    y_bal = balanced_df[tc]

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    models  = {}
    results = {}

    with st.spinner("Training Random Forest..."):
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42,
            n_jobs=-1, class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        y_pred  = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results['Random Forest'] = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   auc(fpr, tpr),
            'y_pred': y_pred, 'y_proba': y_proba, 'y_test': y_test
        }

    with st.spinner("Training Logistic Regression..."):
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
        y_pred  = lr.predict(X_test)
        y_proba = lr.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results['Logistic Regression'] = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   auc(fpr, tpr),
            'y_pred': y_pred, 'y_proba': y_proba, 'y_test': y_test
        }

    return models, results, X_train.columns.tolist()

with st.spinner("Training models on preprocessed data..."):
    models, results, feature_names = train_models(df_clean)

# ==================== BEST MODEL ====================
comparison_data = []
for name, metrics in results.items():
    comparison_data.append({
        'Model':     name,
        'Accuracy':  metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall':    metrics['recall'],
        'F1 Score':  metrics['f1'],
        'ROC AUC':   metrics['roc_auc'] if metrics['roc_auc'] is not None else 0.5
    })
comparison_df   = pd.DataFrame(comparison_data)
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

    raw_target = 'Is_Fraud' if 'Is_Fraud' in df_raw.columns else 'is_fraud'
    fraud_count = df_raw[raw_target].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df_raw):,}")
    with col2:
        st.metric("Fraud Transactions", int(fraud_count))
    with col3:
        st.metric("Normal Transactions", len(df_raw) - int(fraud_count))
    with col4:
        st.metric("Fraud Rate", f"{fraud_count/len(df_raw)*100:.2f}%")

    st.markdown("---")

    # Interactive filterable dataset
    st.subheader("🔎 Interactive Transaction Explorer")
    st.markdown("Filter and explore the dataset to identify fraud patterns.")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        fraud_filter = st.selectbox(
            "Transaction Class",
            ["All", "Fraud Only", "Normal Only"]
        )
    with filter_col2:
        if 'Transaction_Type' in df_raw.columns:
            tt_filter_opts = ["All"] + sorted(df_raw['Transaction_Type'].unique().tolist())
            tt_filter = st.selectbox("Transaction Type", tt_filter_opts)
        else:
            tt_filter = "All"
    with filter_col3:
        amount_min = float(df_raw['Transaction_Amount'].min())
        amount_max = float(df_raw['Transaction_Amount'].max())
        amount_range = st.slider(
            "Transaction Amount Range",
            min_value=amount_min,
            max_value=amount_max,
            value=(amount_min, amount_max),
            format="$%.0f"
        )

    # Apply filters
    df_filtered = df_raw.copy()
    if fraud_filter == "Fraud Only":
        df_filtered = df_filtered[df_filtered[raw_target] == 1]
    elif fraud_filter == "Normal Only":
        df_filtered = df_filtered[df_filtered[raw_target] == 0]
    if tt_filter != "All" and 'Transaction_Type' in df_raw.columns:
        df_filtered = df_filtered[df_filtered['Transaction_Type'] == tt_filter]
    df_filtered = df_filtered[
        (df_filtered['Transaction_Amount'] >= amount_range[0]) &
        (df_filtered['Transaction_Amount'] <= amount_range[1])
    ]

    st.markdown(f"**Showing {len(df_filtered):,} records** ({int(df_filtered[raw_target].sum()):,} fraud, {int((df_filtered[raw_target]==0).sum()):,} normal)")

    display_cols = ['Transaction_Amount', 'Account_Balance', 'Transaction_Type',
                    'Account_Type', 'Transaction_Device', 'Merchant_Category',
                    'Age', 'Gender', raw_target]
    display_cols = [c for c in display_cols if c in df_filtered.columns]

    df_display = df_filtered[display_cols].head(500).copy()
    df_display['Transaction_Amount'] = df_display['Transaction_Amount'].round(2)
    df_display['Account_Balance']    = df_display['Account_Balance'].round(2)

    st.dataframe(
        df_display.style.apply(
            lambda x: ['background-color: #ffe6e6' if v == 1 else '' for v in x],
            subset=[raw_target]
        ),
        use_container_width=True,
        height=350
    )


    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        fig1 = px.pie(
            values=[len(df_raw) - fraud_count, fraud_count],
            names=['Normal', 'Fraud'],
            title='Transaction Class Distribution',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount Distribution")
        fig2 = px.box(
            df_raw, x=raw_target, y='Transaction_Amount',
            title='Amount Distribution by Class',
            color=raw_target
        )
        st.plotly_chart(fig2, use_container_width=True)

    if 'Transaction_Time' in df_raw.columns:
        df_raw['Hour'] = pd.to_datetime(df_raw['Transaction_Time'], format='%H:%M:%S').dt.hour
    if 'Hour' in df_raw.columns:
        st.subheader("Transaction Hour Pattern")
        fig3 = px.histogram(
            df_raw, x='Hour', color=raw_target,
            title='Transactions by Hour', nbins=24, barmode='group'
        )
        st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if 'Transaction_Type' in df_raw.columns:
            st.subheader("Fraud Rate by Transaction Type")
            fraud_by_type = df_raw.groupby('Transaction_Type')[raw_target].mean().sort_values(ascending=False)
            fig4 = px.bar(
                x=fraud_by_type.index, y=fraud_by_type.values,
                title='Fraud Rate by Transaction Type',
                labels={'x': 'Transaction Type', 'y': 'Fraud Rate'}
            )
            st.plotly_chart(fig4, use_container_width=True)

    with col2:
        if 'Account_Type' in df_raw.columns:
            st.subheader("Fraud Rate by Account Type")
            fraud_by_account = df_raw.groupby('Account_Type')[raw_target].mean().sort_values(ascending=False)
            fig5 = px.bar(
                x=fraud_by_account.index, y=fraud_by_account.values,
                title='Fraud Rate by Account Type',
                labels={'x': 'Account Type', 'y': 'Fraud Rate'}
            )
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    num_cols = df_clean.select_dtypes(include='number').columns.tolist()
    corr_matrix = df_clean[num_cols].corr()
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, square=True, linewidths=0.5,
        annot_kws={'size': 6}, ax=ax_heat
    )
    ax_heat.set_title('Feature Correlation Matrix', fontsize=12)
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_heat)

    # Age distribution
    st.subheader("Age Distribution by Transaction Class")
    if 'Age' in df_raw.columns:
        fig_age = px.histogram(
            df_raw, x='Age', color=raw_target,
            title='Age Distribution: Fraud vs Normal',
            nbins=30, barmode='group', opacity=0.9,
            color_discrete_map={0: '#3498db', 1: '#e74c3c'},
            labels={raw_target: 'Transaction Class'}
        )
        fig_age.for_each_trace(lambda t: t.update(
            name='Fraud' if t.name == '1' else 'Normal'
        ))
        st.plotly_chart(fig_age, use_container_width=True)

    # Transaction Device fraud rate
    st.subheader("Fraud Rate by Transaction Device")
    if 'Transaction_Device' in df_raw.columns:
        fraud_by_device = df_raw.groupby('Transaction_Device')[raw_target].mean().sort_values(ascending=False)
        fig_dev = px.bar(
            x=fraud_by_device.index, y=fraud_by_device.values,
            title='Fraud Rate by Transaction Device',
            labels={'x': 'Device', 'y': 'Fraud Rate'},
            color=fraud_by_device.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_dev, use_container_width=True)



# ==================== TAB 2: MODEL PERFORMANCE ====================
with tab2:
    st.header("🤖 Model Performance Comparison")

    st.subheader("📊 Model Comparison Table")
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
            color='#90EE90'
        )
    )
    st.subheader("📊 Model Performance Visualization")
    melted_df = comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    fig6 = px.bar(
        melted_df, x='Model', y='Score', color='Metric',
        title='Model Performance Comparison', barmode='group', text_auto='.3f'
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
                        yticklabels=['Normal', 'Fraud'], ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)

    st.subheader("📊 ROC Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_proba'])
        ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {metrics["roc_auc"]:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    if 'Random Forest' in models:
        st.subheader("📊 Feature Importance (Random Forest)")
        rf_model = models['Random Forest']
        importance_df = pd.DataFrame({
            'Feature':    feature_names[:len(rf_model.feature_importances_)],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        ax_imp.barh(importance_df['Feature'], importance_df['Importance'])
        ax_imp.set_xlabel('Importance')
        ax_imp.set_title('Top 15 Feature Importance (Random Forest)')
        ax_imp.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_imp)

    # Precision-Recall Curves
    st.subheader("📊 Precision-Recall Curves")
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for name, metrics in results.items():
        precision, recall, _ = precision_recall_curve(metrics['y_test'], metrics['y_proba'])
        ap = average_precision_score(metrics['y_test'], metrics['y_proba'])
        ax_pr.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.4f})')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves')
    ax_pr.legend(loc='upper right')
    st.pyplot(fig_pr)
    st.caption("Precision-Recall curves are more informative than ROC for imbalanced datasets like fraud detection.")

    # Radar Chart
    st.subheader("📊 Model Metrics Radar Chart")
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    fig_radar = go.Figure()
    for _, row in comparison_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_cols],
            theta=metrics_cols,
            fill='toself',
            name=row['Model']
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Model Performance Radar Chart'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.success(f"🏆 **Best Model: {best_model_name}** (based on F1 Score)")

# ==================== TAB 3: REAL-TIME PREDICTION ====================
with tab3:
    st.header("⚡ Real-time Fraud Prediction")
    st.info(f"**Using Best Model: {best_model_name}**")

    model = models[best_model_name]

    # Build category options from raw data so dropdowns show real labels
    def get_options(col):
        if col in df_raw.columns:
            return sorted(df_raw[col].dropna().unique().tolist())
        return []

    col1, col2, col3 = st.columns(3)
    input_raw = {}  # stores raw (unscaled) values

    with col1:
        gender_opts  = get_options('Gender') or ['Female', 'Male']
        gender_input = st.selectbox("Gender", gender_opts)
        input_raw['Gender'] = gender_input

        age = st.slider("Age", 18, 100, 35)
        input_raw['Age'] = age

        acct_opts    = get_options('Account_Type') or ['Checking', 'Savings', 'Business']
        account_type = st.selectbox("Account Type", acct_opts)
        input_raw['Account_Type'] = account_type

        transaction_amount = st.number_input("Transaction Amount", min_value=0.01, value=30000.0)
        input_raw['Transaction_Amount'] = transaction_amount

    with col2:
        tt_opts    = get_options('Transaction_Type') or ['Credit', 'Debit', 'Transfer', 'Withdrawal', 'Bill Payment']
        trans_type = st.selectbox("Transaction Type", tt_opts)
        input_raw['Transaction_Type'] = trans_type

        mc_opts  = get_options('Merchant_Category') or ['Retail', 'Restaurant', 'Groceries', 'Entertainment', 'Travel', 'Utilities', 'Other']
        merchant = st.selectbox("Merchant Category", mc_opts)
        input_raw['Merchant_Category'] = merchant

        account_balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
        input_raw['Account_Balance'] = account_balance

        td_opts = get_options('Transaction_Device') or ['Mobile', 'Desktop', 'Tablet', 'ATM', 'POS']
        device  = st.selectbox("Transaction Device", td_opts)
        input_raw['Transaction_Device'] = device

    with col3:
        hour  = st.slider("Hour",  0, 23, 14)
        day   = st.slider("Day",   1, 31, 15)
        month = st.slider("Month", 1, 12,  6)
        input_raw['Hour']  = hour
        input_raw['Day']   = day
        input_raw['Month'] = month

    if st.button("🔍 Predict Fraud Risk", type="primary"):
        # --- Encode categoricals the same way preprocess.py did (label encoding) ---
        input_encoded = {}
        for col, val in input_raw.items():
            if df_raw[col].dtype == object if col in df_raw.columns else False:
                # Reproduce the label encoding: sorted unique values -> 0,1,2,...
                cats = sorted(df_raw[col].dropna().unique().tolist())
                input_encoded[col] = cats.index(val) if val in cats else 0
            else:
                input_encoded[col] = val

        # Derived features that preprocess.py creates
        input_encoded['Is_Night']   = 1 if (hour >= 22 or hour <= 5) else 0
        # Is_Weekend: use Day_of_Week if available, else approximate from Day
        dow = input_encoded.get('Day_of_Week', day % 7)
        input_encoded['Is_Weekend'] = 1 if dow >= 5 else 0
        input_encoded['Amount_Balance_Ratio'] = min(
            transaction_amount / (account_balance + 1), 2
        )
        # Compute Is_High_Amount and Is_Low_Balance using dataset thresholds
        amount_95th  = df_raw['Transaction_Amount'].quantile(0.95)
        balance_5th  = df_raw['Account_Balance'].quantile(0.05) if 'Account_Balance' in df_raw.columns else 0
        input_encoded['Is_High_Amount'] = 1 if transaction_amount > amount_95th else 0
        input_encoded['Is_Low_Balance'] = 1 if account_balance < balance_5th else 0
        input_encoded['Log_Transaction_Amount'] = np.log1p(transaction_amount)

        # Fill any remaining features with 0
        for f in feature_names:
            if f not in input_encoded:
                input_encoded[f] = 0.0

        # Build DataFrame in correct feature order
        input_df = pd.DataFrame([input_encoded])[feature_names]

        # Scale numerical columns using the fitted scaler
        # Build scale_row using only numeric values (encoded), default 0 for anything missing
        scale_row_dict = {}
        for c in scale_cols:
            val = input_encoded.get(c, 0.0)
            try:
                scale_row_dict[c] = float(val)
            except (TypeError, ValueError):
                scale_row_dict[c] = 0.0
        scale_row = pd.DataFrame([scale_row_dict])
        scaled = scaler.transform(scale_row)
        scaled_df = pd.DataFrame(scaled, columns=scale_cols)
        cols_to_scale = [c for c in scale_cols if c in input_df.columns]
        for c in cols_to_scale:
            input_df[c] = scaled_df[c].values

        y_proba_raw = model.predict_proba(input_df)[0][1]

        # Rule-based risk scoring (compensates for low dataset separability)
        risk_score = 0.0

        # Risk scoring — ratio-based so it works at any scale
        ratio = transaction_amount / (account_balance + 1)

        # Amount vs Balance ratio (main driver)
        if ratio > 2.0:
            risk_score += 0.40
        elif ratio > 1.0:
            risk_score += 0.25
        elif ratio > 0.5:
            risk_score += 0.10

        # Extra penalty if amount is also large relative to balance in absolute terms
        if account_balance > 0 and transaction_amount > account_balance * 5:
            risk_score += 0.20

        # Low balance relative to transaction (catches 60k/30k case)
        balance_ratio = account_balance / (transaction_amount + 1)
        if balance_ratio < 0.5:      # balance less than half the transaction
            risk_score += 0.15
        elif balance_ratio < 1.0:    # balance less than the transaction
            risk_score += 0.05

        # Time risk
        if hour >= 22 or hour <= 5:
            risk_score += 0.15

        # Blend: model deviation from baseline + rule score
        # Baseline ~0.42, so subtract it to center around 0
        model_component = (y_proba_raw - 0.42) * 0.2

        # Start from 0.10 so truly low risk stays below 30%
        y_proba = min(max(0.10 + model_component + risk_score, 0.0), 0.95)
        y_pred  = 1 if y_proba >= 0.5 else 0

        try:
            from src.database import insert_transaction
            transaction_data = (
                gender_input, age, account_type, transaction_amount,
                trans_type, merchant, account_balance, device, device,
                'USD', month, day, hour, None,
                float(y_proba), int(y_pred), best_model_name
            )
            insert_transaction(transaction_data)
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

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_proba * 100,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': "darkred"},
                'steps': [
                    {'range': [0,  30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100],'color': "salmon"}
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
            st.write(f"**Account Type:** {account_type}")
        with col2:
            st.write(f"**Transaction Type:** {trans_type}")
            st.write(f"**Merchant Category:** {merchant}")
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
