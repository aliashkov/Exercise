from audioop import reverse
import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

fingerCount = 0
counter = 0
reverseDirection = False
nextFinger = False
stage = "Too far from camera or incorrect position"
fingerName = ""

def calculate_radius(firstDotX , firstDotY , secondDotX , secondDotY):
  xThumb = abs(firstDotX- secondDotX)
  yThumb = abs(firstDotY - secondDotY)
  rThumb = pow(xThumb,2) + pow(yThumb,2)
  return rThumb


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        max_num_hands=1,
        min_tracking_confidence=0.75) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initially set finger count to 0 for each cap
    

    

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        # Get hand index to check label (left or right)
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label
        

        # Set variable to keep landmarks positions (x and y)
        handLandmarks = []

        # Fill list with x and y positions of each landmark
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])
          
        # Get distance between camera and your hand
        distanceWristMiddleFinglerMCP = abs(handLandmarks[0][1] - handLandmarks[9][1])
        distanceIndexFinger =  abs(handLandmarks[17][0] - handLandmarks[5][0])
        
        
        # Calculate if your dot includes in circle area
        
        rThumb = calculate_radius(handLandmarks[4][0], handLandmarks[4][1], (handLandmarks[3][0] * 3 + handLandmarks[4][0]) / 4, (handLandmarks[3][1] * 3  + handLandmarks[4][1]) / 4)
        
        centerDotX = handLandmarks[4][0]
        
        centerDotY = handLandmarks[4][1]
         
        
        fingerX = handLandmarks[8 + fingerCount * 4][0]
        
        fingerY = handLandmarks[8 + fingerCount * 4][1]
        
        
        # Check distance and correct position of your hand
        if (handLandmarks[0][1] < 1 and handLandmarks[0][1] > 0) \
          and (handLandmarks[0][0] < 1 and handLandmarks[0][0] > 0) \
            and (handLandmarks[4][1] < 1 and handLandmarks[4][1] > 0) \
              and (handLandmarks[4][0] < 1 and handLandmarks[4][0] > 0) and distanceWristMiddleFinglerMCP > 0.25:
                if (handLabel == "Left" and handLandmarks[1][0] > handLandmarks[0][0] \
                  and handLandmarks[17][0]  < handLandmarks[1][0] and counter < 5 \
                    and distanceWristMiddleFinglerMCP / 6 < distanceIndexFinger) or \
                      (handLabel == "Right" and handLandmarks[1][0]  <  handLandmarks[0][0] \
                        and handLandmarks[17][0]  > handLandmarks[1][0] and counter >= 5 \
                          and distanceWristMiddleFinglerMCP / 6 < distanceIndexFinger):
                            stage = "Correct"
                            
                            # Check if dot includes in your circle area increase fingerCount
                            if pow(fingerX - centerDotX, 2) + pow(fingerY - centerDotY, 2) <= rThumb:
                              if (fingerCount < 3 and handLandmarks[12 + fingerCount * 4][1] < handLandmarks[11 + fingerCount * 4][1] \
                                and reverseDirection == False) or  \
                                  (fingerCount > 0 and handLandmarks[4 + fingerCount * 4][1] < handLandmarks[3 + fingerCount * 4][1] \
                                    and reverseDirection == True):
                                      if (not reverseDirection):
                                        fingerCount+= 1
                                      else:
                                        fingerCount-= 1
                else:
                  if (counter < 5):
                     stage = "Put your left hand in correct position"
                  else:
                     stage = "Put your right hand in correct position"
        else:
          stage = "Too far from camera or incorrect position"
          
        # Change FingerDirection
        if (fingerCount == 3 and not reverseDirection) or (fingerCount == 0 and reverseDirection):
          if (fingerCount == 0 and reverseDirection):
            nextFinger = not nextFinger
          reverseDirection = not reverseDirection
        
        # If fingers made all circle increase exerciseCounyer
        if (fingerCount == 1 and nextFinger and not reverseDirection):
          nextFinger = not nextFinger
          counter += 1 
          if (counter == 5):
            fingerCount = 0                 

        # Draw hand landmarks 
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Display finger count
    
    cv2.rectangle(image, (0, 0), (550, 73), (245, 117, 16), -1)
    
    cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(image, 'STAGE', (115, 12),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    if (stage == "Correct"):
        stageFingers = "Connect your thumb with "
        match fingerCount:
            case 0:
              fingerName = "Index finger"
            case 1:
              fingerName = "Middle finger"
            case 2:
              fingerName = "Ring finger"
            case 3:
              fingerName = "Pinky finger"
          
        cv2.putText(image, stageFingers + fingerName,
                        (115, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, stage,
                        (115, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
      

    # Display image
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()