#!/usr/bin/env python
# coding: utf-8

# # Facial Emotion Detection using CNN
# 
# This script implements a Convolutional Neural Network (CNN) for facial emotion detection using the FER2013 dataset.

# ## 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ## 2. Load and Explore the Dataset

# Load the FER2013 dataset
data = pd.read_csv('fer2013.csv')
print(data.head())

# Check the shape of the dataset
print(f"Dataset shape: {data.shape}")

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check the distribution of emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
data['emotion_label'] = data['emotion'].map(emotion_labels)

plt.figure(figsize=(10, 6))
sns.countplot(x='emotion_label', data=data)
plt.title('Distribution of Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Print the count of each emotion
print(data['emotion_label'].value_counts())

# ## 3. Data Preprocessing

# Function to preprocess the pixel data
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
    
    return X, y

# Split the data into training, validation, and test sets
train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']
test_data = data[data['Usage'] == 'PrivateTest']

print(f"Training set size: {train_data.shape[0]}")
print(f"Validation set size: {val_data.shape[0]}")
print(f"Test set size: {test_data.shape[0]}")

# Preprocess the data
X_train, y_train = preprocess_data(train_data)
X_val, y_val = preprocess_data(val_data)
X_test, y_test = preprocess_data(test_data)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ## 4. Visualize Sample Images

# Visualize some sample images
plt.figure(figsize=(15, 10))
for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.imshow(X_train[i].reshape(48, 48), cmap='gray')
    plt.title(emotion_labels[np.argmax(y_train[i])])
    plt.axis('off')
plt.tight_layout()
plt.show()

# ## 5. Data Augmentation

# Create data generator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# ## 6. Build CNN Model

# Build a CNN model
def build_model():
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Convolutional Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third Convolutional Block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create the model
model = build_model()
model.summary()

# ## 7. Train the Model

# Define callbacks
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    verbose=1,
    restore_best_weights=True
)

callbacks = [checkpoint, reduce_lr, early_stopping]

# Train the model with data augmentation
batch_size = 64
epochs = 50

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# ## 8. Evaluate the Model

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Load the best model
model.load_weights('best_model.keras')

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=list(emotion_labels.values())))

# Confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(emotion_labels.values()),
            yticklabels=list(emotion_labels.values()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# ## 9. Visualize Predictions

# Visualize some predictions
def plot_predictions(X, y_true, y_pred, n=16):
    plt.figure(figsize=(15, 10))
    for i in range(n):
        plt.subplot(4, 4, i+1)
        plt.imshow(X[i].reshape(48, 48), cmap='gray')
        true_label = emotion_labels[np.argmax(y_true[i])]
        pred_label = emotion_labels[np.argmax(y_pred[i])]
        title = f"True: {true_label}\nPred: {pred_label}"
        if true_label == pred_label:
            color = 'green'
        else:
            color = 'red'
        plt.title(title, color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Get random indices from test set
indices = np.random.randint(0, len(X_test), 16)
plot_predictions(X_test[indices], y_test[indices], y_pred[indices])

# ## 10. Save the Model

# Save the model
model.save('facial_emotion_model.keras')
print("Model saved to 'facial_emotion_model.keras'")

# ## 11. Test with Custom Images (Optional)

# Function to preprocess a custom image
def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path, 0)  # Read as grayscale
    
    # Resize to 48x48
    img = cv2.resize(img, (48, 48))
    
    # Normalize
    img = img / 255.0
    
    # Reshape for model input
    img = img.reshape(1, 48, 48, 1)
    
    return img

# Function to predict emotion
def predict_emotion(image_path):
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(img)[0]
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]
    confidence = prediction[emotion_idx] * 100
    
    # Display image and prediction
    img_display = cv2.imread(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Emotion: {emotion}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()
    
    # Print all emotion probabilities
    for i, emotion_name in emotion_labels.items():
        print(f"{emotion_name}: {prediction[i]*100:.2f}%")

# Example usage (uncomment when you have a custom image)
# predict_emotion('path_to_your_image.jpg')

# ## 12. Conclusion and Next Steps
# 
# In this script, we've built a CNN model for facial emotion detection using the FER2013 dataset. 
# The model architecture includes multiple convolutional layers with batch normalization and dropout for regularization.
# 
# To improve the model further, consider:
# 
# 1. **More complex architectures**: Try deeper networks or pre-trained models like VGG, ResNet, or EfficientNet.
# 2. **Advanced data augmentation**: Implement more sophisticated augmentation techniques.
# 3. **Class imbalance handling**: Address the imbalance in the dataset using techniques like class weights or oversampling.
# 4. **Hyperparameter tuning**: Experiment with different learning rates, batch sizes, and optimizers.
# 5. **Ensemble methods**: Combine multiple models for better performance.
# 6. **Transfer learning**: Use pre-trained models on larger face datasets and fine-tune on FER2013.
# 
# Remember that achieving over 90% accuracy on this dataset is challenging, as even human accuracy on FER2013 is estimated to be around 65-70%. 