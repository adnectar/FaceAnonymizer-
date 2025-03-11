import cv2
import mediapipe as mp
import numpy as np

# Open webcam
video_capture = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75, 0)

while True:
    ret, webcam = video_capture.read()
    
    # Check if frame is captured properly
    if not ret:
        print("Failed to capture frame")
        break
    
    webcamRGB = cv2.cvtColor(webcam, cv2.COLOR_BGR2RGB)
    
    results = faceDetection.process(webcamRGB)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = webcam.shape

            # For blur box
            bbox_x = int(bboxC.xmin * iw)
            bbox_y = int(bboxC.ymin * ih)
            width = int(bboxC.width * iw)
            height = int(bboxC.height * ih)
            
            if width > 0 and height > 0:
                face_roi = webcam[bbox_y:bbox_y + height, bbox_x:bbox_x + width]
            
            blur = cv2.medianBlur(face_roi, 85)
            
            webcam[bbox_y:bbox_y + height, bbox_x:bbox_x + width] = blur

            # For rectange
            cv2.rectangle(webcam, (bbox_x, bbox_y), (bbox_x + width, bbox_y + height), (0,0,255), 3)

    cv2.imshow("Video", webcam)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
video_capture.release()
cv2.destroyAllWindows()
