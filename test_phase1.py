from data_loader import MNISTLoader
import numpy as np
import os

def validate_data():
    print("Starting Phase 1 Validation...")
    loader = MNISTLoader()
    (x_train, y_train), (x_test, y_test) = loader.load()

    print(f"Train Images: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Test Images:  {x_test.shape}, Labels: {y_test.shape}")

    assert x_train.shape == (60000, 784), f"Expected (60000, 784), got {x_train.shape}"
    assert y_train.shape == (60000,), f"Expected (60000,), got {y_train.shape}"
    assert np.max(x_train) <= 1.0 and np.min(x_train) >= 0.0, "Normalization failed"
    
    # Check a few labels
    unique_labels = np.unique(y_train)
    assert len(unique_labels) == 10, f"Expected 10 unique labels, got {len(unique_labels)}"
    
    print("✅ Phase 1 Validation Successful: Data loaded and normalized correctly.")

if __name__ == "__main__":
    try:
        validate_data()
    except Exception as e:
        print(f"❌ Phase 1 Validation Failed: {e}")
        exit(1)
