import cv2
import mediapipe as mp
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the known image
known_image = cv2.imread("deep_18.jpeg")
if known_image is None:
    print("Error: Image not found.")
    exit()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Flag to track if "deep_18" is detected
deep_detected = False

# Function to speak the message
def speak(text):
    engine.say(text)
    engine.runAndWait()

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Detection
        results = face_detection.process(rgb_frame)

        # Draw face detections on the frame
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
                
                # Check if the detected face matches the known image
                if detection.label_id[0] == 0 and not deep_detected:
                    deep_detected = True
                    print("deep_18 is detected")
                    
                    # Speak the message
                    speak("deep_18 is detected")

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
