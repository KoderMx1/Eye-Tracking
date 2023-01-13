import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a Haar cascade classifier to detect eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Read a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray)

    # Draw a rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Eye Tracker', frame)

    # Check for user input
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to quit
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()