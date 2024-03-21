import cv2
import numpy as np

#creating facecascade
face_cascade = cv2.CascadeClassifier("F:\Wrinkles_detection\haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=10)
    
    for x, y, w, h in faces:
        # Crop the region of interest containing the face
        cropped_img = frame[y:y+h, x:x+w]
        
        # Apply Canny edge detection to the cropped region
        edges = cv2.Canny(cropped_img, 130, 1000)
        
        # Count the number of edges detected
        number_of_edges = np.count_nonzero(edges)
        
        # Print the result based on the number of edges
        if number_of_edges > 1000:
            cv2.putText(frame, "Wrinkle Found", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Wrinkle Found", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the frame with detected wrinkles
    cv2.imshow("Wrinkle Detection", frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
