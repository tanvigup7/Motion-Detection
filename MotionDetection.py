import mediapipe as mp
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import imutils


# Function to preprocess frames
def preprocess_frame(frame):
   frame = cv2.resize(frame, (640, 480)) # resize for consistency
   gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale
   gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0) # blur and reduce noise
   return gray_frame


# Initialize video capture
cap = cv2.VideoCapture(1)  # Use 1 for webcam


# Read the first frame to use as the reference
ret, frame = cap.read()
if not ret:
   print("Error: Unable to read from video source.")
   cap.release()
   cv2.destroyAllWindows()
   exit()


# Preprocess the initial frame
reference_frame = preprocess_frame(frame)


while True:
   # Capture the current frame
       ret, frame = cap.read()
       if not ret:
           break


   # Preprocess the current frame
       current_frame = preprocess_frame(frame)


   # Compute the absolute difference between the current frame and the reference frame
       frame_diff = cv2.absdiff(reference_frame, current_frame)


   # Apply a binary threshold to the difference
       _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)


   # Find contours in the thresholded image
       contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       contours = imutils.grab_contours(contours)


   # Draw bounding boxes around detected movements
       for contour in contours:
           if cv2.contourArea(contour) < 400:  # Ignore small movements
               continue
           (x, y, w, h) = cv2.boundingRect(contour)
           cv2.rectangle(frame, (x + 120, y), (x + w + 120, y + h), (0, 255, 0), 2)
          
   # Display the results
       cv2.imshow("Original Frame", frame)
   # cv2.imshow("Frame Difference", frame_diff)
   # cv2.imshow("Thresholded Difference", thresh)


   # Update the reference frame
       reference_frame = current_frame


   # Break loop on 'q' key press
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break


# Release video capture and close windows
cap.release()
cv2.destroyALLWindows()


class landmarker_and_result():
  def __init__(self):
     self.result = mp.tasks.vision.HandLandmarkerResult
     self.landmarker = mp.tasks.vision.HandLandmarker
     self.createLandmarker()
 
  def createLandmarker(self):
     # callback function
     def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result


     # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
     options = mp.tasks.vision.HandLandmarkerOptions(
        base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
        running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
        num_hands = 2, # track both hands
        min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
        min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
        min_tracking_confidence = 0.3, # lower than value to get predictions more often
        result_callback=update_result)
    
     # initialize landmarker
     self.landmarker = self.landmarker.create_from_options(options)
 
  def detect_async(self, frame):
     # convert np frame to mp image
     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
     # detect landmarks
     self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))


  def close(self):
   self.landmarker.close()






