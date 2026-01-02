from ml_utils import train_rf_model
import logging

if __name__ == "__main__":
    # Configure logging to show output in console
    logging.basicConfig(level=logging.INFO)
    print("Training Random Forest Model...")
    train_rf_model("training_data.csv")
