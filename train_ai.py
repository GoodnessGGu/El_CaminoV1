from ml_utils import train_model
from ml_lstm import train_lstm
from settings import config

if __name__ == "__main__":
    if config.model_type == "LSTM":
        train_lstm()
    else:
        train_model()
