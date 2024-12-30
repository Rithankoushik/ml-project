import cv2
import mediapipe as mp

# Initialize mediapipe pose solution
mpDraw = mp.solutions.drawing_utils 
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Open the video file (update the path to the video)
cap = cv2.VideoCapture("E:/ml/gymcheast.mp4")
up = False
counter = 0

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Main loop to read frames from the video
while True:
    success, img = cap.read()
    
    # Break the loop if no more frames are available
    if not success:
        print("Failed to read frame or end of video reached.")
        break

    # Resize the frame to a fixed resolution
    img = cv2.resize(img, (1280, 720))
    
    # Convert the image from BGR to RGB (required by mediapipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect the pose
    results = pose.process(imgRGB)

    # If pose landmarks are detected, draw them on the frame
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        points = {}
        
        # Extract landmark positions
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[id] = (cx, cy)

        # Draw circles on selected body landmarks
        cv2.circle(img, points[12], 15, (255, 0, 0), cv2.FILLED)  # Right shoulder
        cv2.circle(img, points[14], 15, (255, 0, 0), cv2.FILLED)  # Right elbow
        cv2.circle(img, points[11], 15, (255, 0, 0), cv2.FILLED)  # Left shoulder
        cv2.circle(img, points[13], 15, (255, 0, 0), cv2.FILLED)  # Left elbow

        # Simple logic to detect up and down movement
        if not up and points[14][1] + 40 < points[12][1]:  # Right elbow is above the right shoulder
            print("UP")
            up = True
            counter += 1
        elif points[14][1] > points[12][1]:  # Right elbow is below the right shoulder
            print("Down")
            up = False

    # Display the counter on the frame
    cv2.putText(img, str(counter), (100, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)

    # Show the frame in a window
    cv2.imshow("img", img)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
