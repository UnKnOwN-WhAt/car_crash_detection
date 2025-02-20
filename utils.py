import cv2
import joblib
import numpy as np

# Load the trained model
svm_model = joblib.load('svm_accident_detection_model.pkl')

# Function to extract features from a frame
def extract_features_from_frame(frame, size=(64, 64)):
    frame_resized = cv2.resize(frame, size)
    return frame_resized.flatten()

# Test the model on a video
def test_on_video():
    video_path = 'Videos/test.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess and extract features from the frame
        features = extract_features_from_frame(frame)
        features = features.reshape(1, -1)
        
        # Predict using the model
        prediction = svm_model.predict(features)
        label = "Accident" if prediction[0] == 1 else "Non Accident"
        
        # Display the frame with the prediction
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.imshow("Accident Detection", frame)
        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Test the model
#test_on_video('Videos/test.mp4')
