# ... (import statements and app setup)
from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the emotion recognition model
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
label_map = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# For webcam video streaming
camera = cv2.VideoCapture(0)
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

show_warning = False  # Variable to track whether to show the warning

# ... (import statements and app setup)

# ... (import statements and app setup)

# ... (import statements and app setup)

# ... (import statements and app setup)

# ... (import statements and app setup)


# ... (import statements and app setup)

# ... (import statements and app setup)

# ... (import statements and app setup)

# ... (import statements and app setup)

def generate_frames():
    global show_warning

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            # Reset the warning flag
            show_warning = False
            warning_text = None  # Initialize the warning text

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cropped = gray[y:y+h, x:x+w]
                cropped = cv2.resize(cropped, (64, 64))
                cropped = cropped / 255.0
                cropped = np.expand_dims(cropped, axis=0)

                prediction = emotion_classifier.predict(cropped)[0]
                emotion_label = label_map[np.argmax(prediction)]

                # Display emotion label above the rectangle
                cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if emotion_label == "angry":
                    show_warning = True
                    warning_text = "WARNING: Person is ANGRY!"
                elif emotion_label == "happy":
                    show_warning = True
                    warning_text = "WARNING: Person is HAPPY!"
                elif emotion_label == "neutral":
                    show_warning = True
                    warning_text = "WARNING: Person is NEUTRAL!"

            if show_warning and warning_text is not None:
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = 10
                text_y = 30

                # Extend the white background rectangle to match the length of the black text
                rectangle_width = text_size[0] + 20  # Add some padding
                cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + rectangle_width, text_y), (255, 255, 255), -1)

                # Draw black text on top of white rectangle
                cv2.putText(frame, warning_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
