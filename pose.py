import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# VIDEO FEED
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks and draw skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=5)
                                    )
            
            try:
                landmarks = results.pose_landmarks.landmark
                # Your landmark processing code here...
                # Example: calculating angles, accessing specific landmarks
                # Extract landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], etc.
            except Exception as e:
                print(f"Landmark extraction error: {e}")
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
