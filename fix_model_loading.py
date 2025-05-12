#!/usr/bin/env python
# coding: utf-8

"""
Model Loading Verification Script
This script loads the pre-trained model and verifies its accuracy
without any training or modifications.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
from sklearn.metrics import classification_report

# Emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Function to preprocess data exactly as in the original training
def preprocess_data(data):
    # Extract features (pixel values) and labels (emotions)
    pixels = data['pixels'].tolist()
    emotions = data['emotion'].values
    
    # Convert pixel strings to numpy arrays
    X = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split()]
        face = np.array(face).reshape(48, 48)
        X.append(face)
    
    X = np.array(X)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Reshape for CNN input (add channel dimension)
    X = X.reshape(X.shape[0], 48, 48, 1)
    
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(emotions, num_classes=7)
    
    return X, y, emotions

def main():
    print("=" * 50)
    print("MODEL LOADING VERIFICATION")
    print("=" * 50)
    
    # Check if model file exists
    model_path = "facial_emotion_model.keras"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        data = pd.read_csv('fer2013.csv')
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Split the data
    test_data = data[data['Usage'] == 'PrivateTest']
    print(f"Test set size: {test_data.shape[0]} samples")
    
    # Preprocess the test data
    print("\nPreprocessing test data...")
    X_test, y_test, test_emotions = preprocess_data(test_data)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Try different model loading approaches
    print("\nTrying different model loading approaches:")
    
    # Approach 1: Standard loading
    print("\nApproach 1: Standard loading")
    try:
        model1 = load_model(model_path)
        print("Model loaded successfully.")
        
        # Evaluate
        loss1, acc1 = model1.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {acc1:.4f}")
        print(f"Test loss: {loss1:.4f}")
        
        # Make predictions
        y_pred1 = model1.predict(X_test)
        y_pred_classes1 = np.argmax(y_pred1, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes1, 
                                   target_names=list(emotion_labels.values())))
    except Exception as e:
        print(f"Error with Approach 1: {e}")
    
    # Approach 2: Loading with compile=False
    print("\nApproach 2: Loading with compile=False")
    try:
        model2 = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        
        # Compile manually
        model2.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Evaluate
        loss2, acc2 = model2.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {acc2:.4f}")
        print(f"Test loss: {loss2:.4f}")
        
        # Make predictions
        y_pred2 = model2.predict(X_test)
        y_pred_classes2 = np.argmax(y_pred2, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes2, 
                                   target_names=list(emotion_labels.values())))
    except Exception as e:
        print(f"Error with Approach 2: {e}")
    
    # Approach 3: Custom objects
    print("\nApproach 3: Loading with custom objects")
    try:
        model3 = load_model(model_path, compile=False, custom_objects={})
        print("Model loaded successfully.")
        
        # Compile manually with specific optimizer
        model3.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Evaluate
        loss3, acc3 = model3.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {acc3:.4f}")
        print(f"Test loss: {loss3:.4f}")
        
        # Make predictions
        y_pred3 = model3.predict(X_test)
        y_pred_classes3 = np.argmax(y_pred3, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes3, 
                                   target_names=list(emotion_labels.values())))
    except Exception as e:
        print(f"Error with Approach 3: {e}")
    
    print("\n" + "=" * 50)
    print("VERIFICATION COMPLETE")
    print("=" * 50)
    
    # Summary
    print("\nSummary of model loading approaches:")
    try:
        print(f"Approach 1 accuracy: {acc1:.4f}")
    except:
        print("Approach 1 failed")
    
    try:
        print(f"Approach 2 accuracy: {acc2:.4f}")
    except:
        print("Approach 2 failed")
    
    try:
        print(f"Approach 3 accuracy: {acc3:.4f}")
    except:
        print("Approach 3 failed")
    
    # Recommendation
    print("\nRecommendation:")
    accuracies = []
    approaches = []
    
    try:
        accuracies.append(acc1)
        approaches.append("Approach 1")
    except:
        pass
    
    try:
        accuracies.append(acc2)
        approaches.append("Approach 2")
    except:
        pass
    
    try:
        accuracies.append(acc3)
        approaches.append("Approach 3")
    except:
        pass
    
    if accuracies:
        best_idx = np.argmax(accuracies)
        print(f"The best approach is {approaches[best_idx]} with accuracy {accuracies[best_idx]:.4f}")
        print(f"Use this approach for continued training.")
    else:
        print("All approaches failed. The model file may be corrupted or incompatible.")

if __name__ == "__main__":
    main() 