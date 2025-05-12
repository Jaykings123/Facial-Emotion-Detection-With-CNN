import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from collections import deque
import datetime
import os

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

# Create directory for saving images
os.makedirs("captured_emotions", exist_ok=True)

# Load the pre-trained emotion detection model
try:
    model = load_model("facial_emotion_model.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize emotion history tracking
emotion_history = deque(maxlen=30)  # Track last 30 frames
emotion_counts = {emotion: 0 for emotion in emotion_labels}
dominant_emotion = "None"
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 128),  # Brown
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 255),    # Yellow
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (255, 128, 0), # Cyan
    'Neutral': (255, 255, 255) # White
}

# Initialize webcam (try 0 or cv2.CAP_ANY if issues arise)
cap = cv2.VideoCapture(0)

# Set resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load YuNet face detector
try:
    face_detector = cv2.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 320),
        0.9,  # Score threshold
        0.3,  # NMS threshold
        5000  # Top K
    )
    print("YuNet face detector loaded successfully!")
except Exception as e:
    print(f"Error loading YuNet face detector: {e}")
    print("Falling back to DNN face detector...")
    
    # Try to load OpenCV DNN Face Detector as fallback
    try:
        face_net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt", 
            "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
        use_yunet = False
        print("DNN face detector loaded successfully!")
    except Exception as e:
        print(f"Error loading DNN face detector: {e}")
        exit()
else:
    use_yunet = True

# Application state variables
show_fps = True
apply_filters = False
show_histogram = False
recording = False
record_start_time = None
snapshot_mode = False
show_help = False
last_frame_time = time.time()
fps = 0

# Create a simple filter based on emotion
def apply_emotion_filter(frame, emotion):
    filtered = frame.copy()
    
    if emotion == 'Happy':
        # Bright, warm filter
        filtered = cv2.addWeighted(filtered, 1.2, np.zeros(filtered.shape, filtered.dtype), 0, 10)
        # Add a slight yellow tint
        filtered[:,:,0] = np.clip(filtered[:,:,0] * 0.9, 0, 255)  # Reduce blue
    
    elif emotion == 'Sad':
        # Blue, cold filter
        filtered = cv2.addWeighted(filtered, 0.9, np.zeros(filtered.shape, filtered.dtype), 0, -10)
        # Add blue tint
        filtered[:,:,0] = np.clip(filtered[:,:,0] * 1.2, 0, 255)  # Increase blue
        filtered[:,:,1] = np.clip(filtered[:,:,1] * 0.9, 0, 255)  # Reduce green
        filtered[:,:,2] = np.clip(filtered[:,:,2] * 0.9, 0, 255)  # Reduce red
    
    elif emotion == 'Angry':
        # Red tint
        filtered[:,:,2] = np.clip(filtered[:,:,2] * 1.5, 0, 255)  # Increase red
        filtered = cv2.addWeighted(filtered, 1.1, np.zeros(filtered.shape, filtered.dtype), 0, 0)
    
    elif emotion == 'Surprise':
        # High contrast, bright
        filtered = cv2.addWeighted(filtered, 1.3, np.zeros(filtered.shape, filtered.dtype), 0, 15)
    
    elif emotion == 'Fear':
        # Dark, high contrast
        filtered = cv2.addWeighted(filtered, 1.2, np.zeros(filtered.shape, filtered.dtype), 0, -20)
    
    elif emotion == 'Disgust':
        # Green tint
        filtered[:,:,1] = np.clip(filtered[:,:,1] * 1.2, 0, 255)  # Increase green
        filtered = cv2.addWeighted(filtered, 0.9, np.zeros(filtered.shape, filtered.dtype), 0, -5)
    
    return filtered

# Function to draw emotion histogram
def draw_emotion_histogram(frame, emotion_counts):
    h, w = frame.shape[:2]
    histogram_height = 100
    histogram_width = w
    histogram = np.ones((histogram_height, histogram_width, 3), dtype=np.uint8) * 255
    
    # Find the maximum count for scaling
    max_count = max(emotion_counts.values()) if any(emotion_counts.values()) else 1
    
    # Draw bars
    bar_width = histogram_width // len(emotion_labels)
    for i, emotion in enumerate(emotion_labels):
        count = emotion_counts[emotion]
        bar_height = int((count / max_count) * (histogram_height - 20))
        color = emotion_colors[emotion]
        
        # Draw the bar
        start_x = i * bar_width
        cv2.rectangle(histogram, 
                     (start_x, histogram_height - 10 - bar_height), 
                     (start_x + bar_width - 5, histogram_height - 10), 
                     color, -1)
        
        # Add label
        cv2.putText(histogram, emotion[:3], (start_x + 5, histogram_height - 15 - bar_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Combine with the main frame
    result = np.vstack([frame, histogram])
    return result

# Function to save a snapshot with emotion label
def save_snapshot(frame, emotion):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_emotions/{emotion}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved as {filename}")
    return filename

print("Starting webcam capture. Press 'q' to quit.")
print("Controls:")
print("  H - Show/hide help")
print("  F - Toggle FPS display")
print("  E - Toggle emotion-based filters")
print("  G - Toggle emotion histogram")
print("  S - Take a snapshot")
print("  R - Start/stop recording emotions")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        continue
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - last_frame_time)
    last_frame_time = current_time
    
    # Flip the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)
    
    h, w = frame.shape[:2]
    
    # Create a copy for displaying
    display_frame = frame.copy()
    
    # Face detection and emotion recognition
    faces_detected = False
    current_emotions = []
    
    if use_yunet:
        # Set input size for YuNet detector
        face_detector.setInputSize((w, h))
        
        # Detect faces
        _, faces = face_detector.detect(frame)
        
        if faces is not None:
            faces_detected = True
            for face in faces:
                # Get bounding box coordinates
                x, y, w_box, h_box = face[0:4].astype(np.int32)
                
                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                x_max = min(w, x + w_box)
                y_max = min(h, y + h_box)
                
                # Extract face ROI
                roi = frame[y:y_max, x:x_max]
                if roi.size == 0:
                    continue  # Skip if ROI is empty
                
                # Preprocess for emotion detection
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.reshape(1, 48, 48, 1) / 255.0  # Normalize
                
                # Predict emotion
                pred = model.predict(roi_gray, verbose=0)
                emotion_idx = np.argmax(pred)
                label = emotion_labels[emotion_idx]
                confidence = np.max(pred) * 100
                
                # Add to current emotions
                current_emotions.append(label)
                
                # Get color based on emotion
                color = emotion_colors[label]
                
                # Draw bounding box and label
                cv2.rectangle(display_frame, (x, y), (x_max, y_max), color, 2)
                cv2.putText(display_frame, f"{label} ({confidence:.1f}%)", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # Convert frame to blob for DNN face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        
        # Detect faces
        face_net.setInput(blob)
        detections = face_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Consider faces with confidence > 50%
                faces_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x_max, y_max) = box.astype("int")
                
                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                # Extract face ROI
                roi = frame[y:y_max, x:x_max]
                if roi.size == 0:
                    continue  # Skip if ROI is empty
                
                # Preprocess for emotion detection
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.reshape(1, 48, 48, 1) / 255.0  # Normalize
                
                # Predict emotion
                pred = model.predict(roi_gray, verbose=0)
                emotion_idx = np.argmax(pred)
                label = emotion_labels[emotion_idx]
                confidence = np.max(pred) * 100
                
                # Add to current emotions
                current_emotions.append(label)
                
                # Get color based on emotion
                color = emotion_colors[label]
                
                # Draw bounding box and label
                cv2.rectangle(display_frame, (x, y), (x_max, y_max), color, 2)
                cv2.putText(display_frame, f"{label} ({confidence:.1f}%)", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Update emotion history and counts
    if current_emotions:
        # Add all detected emotions to history
        for emotion in current_emotions:
            emotion_history.append(emotion)
            emotion_counts[emotion] += 1
        
        # Determine dominant emotion
        if emotion_history:
            # Count occurrences of each emotion in recent history
            recent_counts = {}
            for e in emotion_history:
                recent_counts[e] = recent_counts.get(e, 0) + 1
            
            # Find the most common emotion
            dominant_emotion = max(recent_counts, key=recent_counts.get)
    
    # Apply emotion-based filter if enabled
    if apply_filters and dominant_emotion != "None" and faces_detected:
        display_frame = apply_emotion_filter(display_frame, dominant_emotion)
    
    # Take snapshot if in snapshot mode
    if snapshot_mode and faces_detected:
        if current_emotions:
            filename = save_snapshot(frame, current_emotions[0])
            # Display confirmation
            cv2.putText(display_frame, f"Snapshot saved!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        snapshot_mode = False
    
    # Display recording indicator
    if recording:
        # Calculate elapsed time
        elapsed = time.time() - record_start_time
        cv2.putText(display_frame, f"Recording: {int(elapsed)}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display FPS if enabled
    if show_fps:
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display dominant emotion
    if faces_detected:
        cv2.putText(display_frame, f"Dominant: {dominant_emotion}", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_colors.get(dominant_emotion, (255, 255, 255)), 2)
    
    # Show help overlay if enabled
    if show_help:
        help_overlay = display_frame.copy()
        cv2.rectangle(help_overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
        cv2.putText(help_overlay, "CONTROLS:", (w//4 + 20, h//4 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        controls = [
            "H - Show/hide this help",
            "F - Toggle FPS display",
            "E - Toggle emotion filters",
            "G - Toggle emotion histogram",
            "S - Take a snapshot",
            "R - Start/stop recording",
            "Q - Quit application"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(help_overlay, control, (w//4 + 20, h//4 + 70 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Blend the help overlay with the display frame
        alpha = 0.7
        display_frame = cv2.addWeighted(display_frame, 1 - alpha, help_overlay, alpha, 0)
    
    # Add emotion histogram if enabled
    if show_histogram:
        display_frame = draw_emotion_histogram(display_frame, emotion_counts)
    
    # Show frame
    cv2.imshow('Emotion Detection', display_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('h'):
        show_help = not show_help
    elif key == ord('f'):
        show_fps = not show_fps
    elif key == ord('e'):
        apply_filters = not apply_filters
        if apply_filters:
            print("Emotion filters enabled")
        else:
            print("Emotion filters disabled")
    elif key == ord('g'):
        show_histogram = not show_histogram
    elif key == ord('s'):
        snapshot_mode = True
    elif key == ord('r'):
        if not recording:
            recording = True
            record_start_time = time.time()
            print("Recording started")
        else:
            recording = False
            elapsed = time.time() - record_start_time
            print(f"Recording stopped after {elapsed:.1f} seconds")
            
            # Print emotion summary
            print("\nEmotion Summary:")
            total = sum(emotion_counts.values())
            for emotion, count in emotion_counts.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"  {emotion}: {count} ({percentage:.1f}%)")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam capture ended.")