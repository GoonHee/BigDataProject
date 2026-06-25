import os
from dotenv import load_dotenv


load_dotenv()


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'fraud_data.csv')


DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'fraud_detection'),  
    'user': os.getenv('DB_USER', 'postgres'),            
    'password': os.getenv('DB_PASSWORD', 'r2u0x0i5n')   
}

# SQLAlchemy connection string
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Column names
TARGET_COL = 'Is_Fraud'
CATEGORICAL_COLS = ['Gender', 'Account_Type', 'Transaction_Type', 'Merchant_Category', 
                    'Transaction_Device', 'Device_Type', 'Transaction_Currency']
NUMERICAL_COLS = ['Age', 'Transaction_Amount', 'Account_Balance', 'Month', 'Day', 'Hour']

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_detector.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.pkl')