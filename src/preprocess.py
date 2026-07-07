import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded raw data: {len(df):,} records")
        logger.info(f"📋 Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load: {e}")
        raise


def handle_missing_values(df):
    
    df_copy = df.copy()
    
    # Check for missing values
    missing_counts = df_copy.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        logger.info(f"Found missing values: {total_missing}")
        
        # Numerical columns: fill with median
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_copy[col].isnull().sum() > 0:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                logger.debug(f"  Filled {col} with median")
        
        # Categorical columns: fill with mode or 'Unknown'
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_copy[col].isnull().sum() > 0:
                mode_val = df_copy[col].mode()
                if len(mode_val) > 0:
                    df_copy[col] = df_copy[col].fillna(mode_val[0])
                else:
                    df_copy[col] = df_copy[col].fillna('Unknown')
                logger.debug(f"  Filled {col}")
    else:
        logger.info("✅ No missing values found")
    
    return df_copy

def drop_unnecessary_columns(df):
    """Drop columns that should not be used for training"""
    cols_to_drop = [
        'Customer_ID',
        'Customer_Name', 
        'Transaction_ID',
        'Merchant_ID',
        'Customer_Contact',
        'Customer_Email',
        'Transaction_Description',
        'Transaction_Currency',  # 全是 INR
        'State',  # 与 City 重复
        'Bank_Branch',  # 与 City 重复
        'Transaction_Location',  # 与 City 重复
    ]
    
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"  Dropped: {col}")
    
    return df

def create_feature_engineering(df):
    
    df_copy = df.copy()
    
    logger.info("🔧 Starting feature engineering...")
    
    # 1. Amount to Balance Ratio (how much of balance is being spent)
    if 'Transaction_Amount' in df_copy.columns and 'Account_Balance' in df_copy.columns:
        df_copy['Amount_Balance_Ratio'] = df_copy['Transaction_Amount'] / (df_copy['Account_Balance'] + 1)
        df_copy['Amount_Balance_Ratio'] = df_copy['Amount_Balance_Ratio'].clip(upper=2)
        logger.info("  ✅ Created Amount_Balance_Ratio")
    
    # 2. Night Transaction (22:00 - 05:00 has higher risk)
    if 'Hour' in df_copy.columns:
        df_copy['Is_Night'] = ((df_copy['Hour'] >= 22) | (df_copy['Hour'] <= 5)).astype(int)
        logger.info("  ✅ Created Is_Night")
    
    # 3. Weekend Transaction
    if 'Day' in df_copy.columns:
        # If Day is day of week (0-6), use this logic
        if df_copy['Day'].max() <= 6:
            df_copy['Is_Weekend'] = (df_copy['Day'] >= 5).astype(int)
        else:
            # Day is day of month, approximate with modulo 7
            df_copy['Is_Weekend'] = ((df_copy['Day'] % 7) == 0).astype(int)
        logger.info("  ✅ Created Is_Weekend")
    
    # 4. High Amount Flag (above 95th percentile)
    if 'Transaction_Amount' in df_copy.columns:
        amount_95th = df_copy['Transaction_Amount'].quantile(0.95)
        df_copy['Is_High_Amount'] = (df_copy['Transaction_Amount'] > amount_95th).astype(int)
        logger.info("  ✅ Created Is_High_Amount")
    
    # 5. Low Balance Flag (below 5th percentile)
    if 'Account_Balance' in df_copy.columns:
        balance_5th = df_copy['Account_Balance'].quantile(0.05)
        df_copy['Is_Low_Balance'] = (df_copy['Account_Balance'] < balance_5th).astype(int)
        logger.info("  ✅ Created Is_Low_Balance")
    
    # 6. Log Transaction Amount (reduce skewness)
    if 'Transaction_Amount' in df_copy.columns:
        df_copy['Log_Transaction_Amount'] = np.log1p(df_copy['Transaction_Amount'])
        logger.info("  ✅ Created Log_Transaction_Amount")
    
    return df_copy


def encode_categorical_columns(df):
    
    df_copy = df.copy()
    label_encoders = {}
    
    # Identify categorical columns (object type)
    categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"🔧 Encoding {len(categorical_cols)} categorical columns...")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_copy[col] = df_copy[col].astype(str)
        df_copy[col] = le.fit_transform(df_copy[col])
        label_encoders[col] = le
        logger.info(f"  ✅ Encoded {col} -> {len(le.classes_)} categories")
    
    return df_copy, label_encoders


def scale_numerical_columns(df):
    
    df_copy = df.copy()
    
    # Identify numerical columns (excluding binary/boolean columns)
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target column and binary columns (0/1)
    exclude_cols = ['Is_Fraud', 'Is_Night', 'Is_Weekend', 'Is_High_Amount', 'Is_Low_Balance']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    if numerical_cols:
        scaler = StandardScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
        logger.info(f"✅ Scaled {len(numerical_cols)} numerical columns: {numerical_cols}")
        return df_copy, scaler
    
    return df_copy, None


def save_preprocessed_data(df, filepath):
   
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath, index=False)
    logger.info(f"✅ Saved {len(df):,} records to {filepath}")


def load_preprocessed_data(filepath):
   
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded {len(df):,} records from {filepath}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load: {e}")
        raise

def extract_time_features(df):
    """Extract hour, day, month from date/time columns"""
    df_copy = df.copy()
    
    # Process Transaction_Date
    if 'Transaction_Date' in df_copy.columns:
        df_copy['Transaction_Date'] = pd.to_datetime(df_copy['Transaction_Date'], format='%d-%m-%Y')
        df_copy['Day'] = df_copy['Transaction_Date'].dt.day
        df_copy['Month'] = df_copy['Transaction_Date'].dt.month
        df_copy['Day_of_Week'] = df_copy['Transaction_Date'].dt.dayofweek
        df_copy['Is_Weekend'] = (df_copy['Day_of_Week'] >= 5).astype(int)
        
        # Drop original date column
        df_copy = df_copy.drop(columns=['Transaction_Date'])
        logger.info("✅ Extracted Day, Month, Day_of_Week, Is_Weekend from Transaction_Date")
    
    # Process Transaction_Time
    if 'Transaction_Time' in df_copy.columns:
        # Convert to datetime to extract hour
        df_copy['Transaction_Time'] = pd.to_datetime(df_copy['Transaction_Time'], format='%H:%M:%S')
        df_copy['Hour'] = df_copy['Transaction_Time'].dt.hour
        
        # Drop original time column
        df_copy = df_copy.drop(columns=['Transaction_Time'])
        logger.info("✅ Extracted Hour from Transaction_Time")
    
    return df_copy

def print_data_summary(df, title="DATA SUMMARY"):
    """
    Print a summary of the data
    
    Parameters:
    df: DataFrame
    title: Summary title
    """
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    
    print(f"\nTotal records: {len(df):,}")
    
    if 'Is_Fraud' in df.columns:
        fraud_count = df['Is_Fraud'].sum()
        print(f"Fraud cases: {int(fraud_count):,} ({fraud_count/len(df)*100:.2f}%)")
        print(f"Normal cases: {len(df) - int(fraud_count):,}")
    
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ Missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  - {col}: {count}")
    else:
        print("\n✅ No missing values")
    
    print("=" * 60)


def full_preprocessing_pipeline(raw_filepath, output_filepath, do_feature_engineering=True):
    print("\n" + "=" * 60)
    print("🔍 FRAUD DETECTION - DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\n📂 Step 1: Loading raw data...")
    df = load_raw_data(raw_filepath)
    
    # Step 2: Handle missing values
    print("\n🔧 Step 2: Handling missing values...")
    df = handle_missing_values(df)
    
    # === NEW: Extract time features ===
    print("\n🔧 Step 3: Extracting time features...")
    df = extract_time_features(df)
    
    # Step 4: Feature engineering (optional)
    if do_feature_engineering:
        print("\n🔧 Step 4: Creating new features...")
        df = create_feature_engineering(df)
    
    # Step 5: Encode categorical columns
    print("\n🔧 Step 5: Encoding categorical columns...")
    df, label_encoders = encode_categorical_columns(df)
    
    # Step 6: Scale numerical columns
    print("\n🔧 Step 6: Scaling numerical columns...")
    df, scaler = scale_numerical_columns(df)
    
    # Step 7: Save preprocessed data
    print("\n💾 Step 7: Saving preprocessed data...")
    save_preprocessed_data(df, output_filepath)
    
    # Print summary
    print_data_summary(df, "PREPROCESSED DATA SUMMARY")
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    return df


# ==================== MAIN ====================
if __name__ == "__main__":
    # Use project-local paths so the script works on any machine
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "Bank_Transaction_Fraud_Detection.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "data", "preprocessed_data.csv")
    
    # Run preprocessing
    df_processed = full_preprocessing_pipeline(
        raw_filepath=RAW_DATA_PATH,
        output_filepath=OUTPUT_PATH,
        do_feature_engineering=True  # Set to False if you don't want new features
    )
    
    # Verify the saved file
    print("\n🔍 Verifying saved file...")
    df_check = pd.read_csv(OUTPUT_PATH)
    print(f"✅ Verified: {len(df_check)} records, {len(df_check.columns)} columns")