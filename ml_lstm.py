import pandas as pd
import numpy as np
import os
import logging
import joblib

# Optional Imports (to avoid crashing on Py 3.14 if not installed yet)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

SCALER_PATH = os.path.join("models", "scaler.pkl")
MODEL_PATH = os.path.join("models", "lstm_model.h5") # Keras format
SEQ_LENGTH = 10 # Number of past candles to look at

def check_tf():
    if not TF_AVAILABLE:
        logger.error("❌ TensorFlow not installed! Cannot use LSTM.")
        return False
    return True

def prepare_sequences(data, seq_length):
    """
    Converts 2D array [rows, features] into 3D array [samples, seq_length, features]
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length]) # We predict the NEXT step's outcome? 
        # Actually usually we have 'outcome' aligned.
        # Ideally: Train X[0..9] -> Predict Label[10] (or 9 depending on alignment)
        
    return np.array(X)

def train_lstm(data_path="training_data.csv"):
    if not check_tf(): return

    df = pd.read_csv(data_path)
    if 'outcome' not in df.columns:
        return
        
    # 1. Feature Selection (Same as XGBoost)
    drop_raw = ['open', 'close', 'min', 'max', 'volume', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower']
    drop_meta = ['time', 'outcome', 'signal', 'asset', 'from', 'to', 'next_close', 'next_open', 'id', 'at']
    cols_to_drop = [c for c in drop_raw + drop_meta if c in df.columns]
    
    # Features
    feature_data = df.drop(columns=cols_to_drop)
    
    # Target
    targets = df['outcome'].values
    
    # 2. Scaling (Critical for LSTM)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    # Save scaler for inference
    if not os.path.exists("models"): os.makedirs("models")
    joblib.dump(scaler, SCALER_PATH)
    
    # 3. Create Sequences
    # X: [Samples, 10, Features]
    # y: [Samples] (The label of the LAST candle in the sequence? Or the NEXT?)
    # Our 'outcome' in row N is "Did trade at N win?".
    # So we want to use Candles N-9..N to predict Outcome N.
    
    X_seq = []
    y_seq = []
    
    for i in range(SEQ_LENGTH, len(scaled_features)):
        # Sequence: Rows i-10 to i (exclusive end) -> 10 rows
        seq = scaled_features[i-SEQ_LENGTH : i]
        X_seq.append(seq)
        y_seq.append(targets[i-1]) # Outcome of the last candle in sequence (i-1) -> Next candle color (i)
        
    X_train = np.array(X_seq)
    y_train = np.array(y_seq)
    
    logger.info(f"LSTM Training Data: {X_train.shape}")
    
    # 4. Build Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 5. Train
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    model.save(MODEL_PATH)
    logger.info(f"✅ LSTM Model saved to {MODEL_PATH}")


# Global Cache
LOADED_MODEL = None
LOADED_SCALER = None

def get_model_and_scaler():
    global LOADED_MODEL, LOADED_SCALER
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
        
    if LOADED_MODEL is None:
        try:
            logger.info("Loading LSTM Model from disk...")
            LOADED_MODEL = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None, None
            
    if LOADED_SCALER is None:
        try:
            LOADED_SCALER = joblib.load(SCALER_PATH)
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return None, None
            
    return LOADED_MODEL, LOADED_SCALER

def predict_lstm(df_window):
    """
    df_window: DataFrame containing last 10 candles (raw features).
    """
    if not check_tf(): return 0
    
    model, scaler = get_model_and_scaler()
    
    if model is None or scaler is None:
        return 0 # No model
        
    try:
        # Preprocess features same as training
        drop_raw = ['open', 'close', 'min', 'max', 'volume', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower']
        drop_meta = ['time', 'outcome', 'signal', 'asset', 'from', 'to']
        cols_to_drop = [c for c in drop_raw + drop_meta if c in df_window.columns]
        
        features = df_window.drop(columns=cols_to_drop)
        
        # Scale
        scaled = scaler.transform(features)
        
        # Reshape [1, 10, features]
        input_seq = np.array([scaled])
        
        # Predict
        prob = model.predict(input_seq, verbose=0)[0][0]
        
        return 1 if prob > 0.55 else 0
        
    except Exception as e:
        logger.error(f"LSTM Prediction Error: {e}")
        return 0


if __name__ == "__main__":
    train_lstm()
