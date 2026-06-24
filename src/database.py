import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from src.config import DB_CONFIG, DATABASE_URL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_connection():
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],        
            port=DB_CONFIG['port'],        
            database=DB_CONFIG['database'], 
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def get_engine():
    return create_engine(DATABASE_URL)


def create_tables(conn):
    cursor = conn.cursor()
    
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS fraud_transactions (
            id SERIAL PRIMARY KEY,
            
            -- Original transaction data
            gender VARCHAR(10),
            age INTEGER,
            account_type VARCHAR(20),
            transaction_amount DECIMAL(15,2),
            transaction_type VARCHAR(30),
            merchant_category VARCHAR(50),
            account_balance DECIMAL(15,2),
            transaction_device VARCHAR(20),
            device_type VARCHAR(20),
            transaction_currency VARCHAR(5),
            month INTEGER,
            day INTEGER,
            hour INTEGER,
            
            -- Actual label (for historical data)
            is_fraud_actual INTEGER,
            
            -- Prediction results
            fraud_probability DECIMAL(5,4),
            prediction_result INTEGER,
            model_used VARCHAR(50),
            
            -- Metadata
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    cursor.execute(create_table_sql)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON fraud_transactions(prediction_timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_result ON fraud_transactions(prediction_result)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transaction_amount ON fraud_transactions(transaction_amount)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_account_type ON fraud_transactions(account_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hour ON fraud_transactions(hour)")
    
    # Create view for daily statistics
    cursor.execute("""
        CREATE OR REPLACE VIEW daily_fraud_stats AS
        SELECT 
            DATE(prediction_timestamp) as date,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN prediction_result = 1 THEN 1 ELSE 0 END) as fraud_predictions,
            ROUND(AVG(fraud_probability) * 100, 2) as avg_fraud_probability_percent,
            ROUND(100.0 * SUM(CASE WHEN prediction_result = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as fraud_percentage
        FROM fraud_transactions
        WHERE prediction_result IS NOT NULL
        GROUP BY DATE(prediction_timestamp)
        ORDER BY date DESC
    """)
    
    # Create function to get high-risk transactions - FIXED
    cursor.execute("""
        CREATE OR REPLACE FUNCTION get_high_risk_transactions(risk_threshold DECIMAL)
        RETURNS TABLE(
            transaction_id INTEGER,
            amount DECIMAL,
            fraud_probability DECIMAL,
            prediction_time TIMESTAMP
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                id,
                transaction_amount,
                fraud_probability,
                prediction_timestamp
            FROM fraud_transactions
            WHERE fraud_probability >= risk_threshold
            ORDER BY fraud_probability DESC
            LIMIT 100;
        END;
        $$ LANGUAGE plpgsql
    """)
    
    conn.commit()
    cursor.close()
    logger.info("✅ Tables, indexes, views, and functions created successfully")
def init_database():
    """Initialize database with tables (no SQL file needed)"""
    conn = get_connection()
    create_tables(conn)
    conn.close()
    logger.info("✅ Database initialized successfully")
    return True


def insert_transaction(transaction_data):
    """Insert a single transaction prediction"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO fraud_transactions (
            gender, age, account_type, transaction_amount, transaction_type,
            merchant_category, account_balance, transaction_device, device_type,
            transaction_currency, month, day, hour, is_fraud_actual,
            fraud_probability, prediction_result, model_used
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    
    try:
        cursor.execute(query, transaction_data)
        transaction_id = cursor.fetchone()[0]
        conn.commit()
        logger.info(f"✅ Transaction {transaction_id} saved to database")
        return transaction_id
    except Exception as e:
        logger.error(f"Insert failed: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()


def insert_batch_transactions(transactions_df):
    """Insert multiple transactions efficiently"""
    try:
        engine = get_engine()
        transactions_df.to_sql('fraud_transactions', engine, if_exists='append', index=False)
        logger.info(f"✅ Batch inserted {len(transactions_df)} transactions")
        return True
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        return False


def get_transaction_history(limit=100, offset=0):
    """Get recent transaction history with pagination"""
    conn = get_connection()
    
    query = f"""
        SELECT * FROM fraud_transactions 
        ORDER BY prediction_timestamp DESC 
        LIMIT {limit} OFFSET {offset}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_statistics():
    """Get database statistics"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total records
    cursor.execute("SELECT COUNT(*) FROM fraud_transactions")
    total = cursor.fetchone()[0]
    
    # Fraud vs Normal
    cursor.execute("""
        SELECT prediction_result, COUNT(*) 
        FROM fraud_transactions 
        WHERE prediction_result IS NOT NULL
        GROUP BY prediction_result
    """)
    results = cursor.fetchall()
    
    # Daily stats from view
    try:
        cursor.execute("SELECT * FROM daily_fraud_stats LIMIT 7")
        daily_stats = cursor.fetchall()
    except:
        daily_stats = []
    
    cursor.close()
    conn.close()
    
    return {
        'total_records': total,
        'predictions': dict(results),
        'daily_stats': daily_stats
    }


def get_high_risk_transactions(threshold=0.7):
    """Get high-risk transactions using database function"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM get_high_risk_transactions(%s)", (threshold,))
        results = cursor.fetchall()
    except Exception as e:
        logger.error(f"Function call failed: {e}")
        results = []
    
    cursor.close()
    conn.close()
    return results


def update_actual_label(transaction_id, is_fraud_actual):
    """Update actual fraud label after verification"""
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
        UPDATE fraud_transactions 
        SET is_fraud_actual = %s 
        WHERE id = %s
    """
    
    try:
        cursor.execute(query, (is_fraud_actual, transaction_id))
        conn.commit()
        logger.info(f"✅ Updated actual label for transaction {transaction_id}")
        return True
    except Exception as e:
        logger.error(f"Update failed: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def get_model_performance():
    """Calculate model performance based on verified transactions"""
    conn = get_connection()
    
    query = """
        SELECT 
            COUNT(*) as verified_count,
            SUM(CASE WHEN prediction_result = is_fraud_actual THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(AVG(CASE WHEN prediction_result = is_fraud_actual THEN 1.0 ELSE 0.0 END), 4) as accuracy
        FROM fraud_transactions
        WHERE is_fraud_actual IS NOT NULL AND prediction_result IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return {'verified_count': 0, 'correct_predictions': 0, 'accuracy': 0}
    
    return df.iloc[0].to_dict()


def export_to_csv(filepath='exports/fraud_data_export.csv'):
    """Export all data to CSV"""
    os.makedirs('exports', exist_ok=True)
    
    df = get_transaction_history(limit=10000)
    df.to_csv(filepath, index=False)
    logger.info(f"✅ Exported to {filepath}")
    return filepath


def check_connection():
    """Test database connection"""
    try:
        conn = get_connection()
        conn.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False