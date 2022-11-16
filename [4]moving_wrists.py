import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

# Hands Counter
counter = 0
fingerCount = 0
frameCounter = 5

stage = "Too far from camera or incorrect position"

#Left hand position

leftHandFrames = 0
isLeftHandFixating = False
leftHandPosition = 0
leftHandDistanceLevel = 50
isLeftHandDown = False
isLeftHandUp = False

#Right hand position

rightHandFrames = 0
isRightHandFixating = False
rightHandPosition = 0
rightHandDistanceLevel = 50
isRightHandDown = False
isRightHandUp = False

handsDirection = False

def handsChangeDirection(currentLeftHandDistance , currentRightHandDistance):
  
  # Extracting data from current hand position
  
  handsNewDirection = handsDirection
  counterHands = counter
  isRHandUp = isRightHandUp
  isLHandUp = isLeftHandUp
  isLHandDown = isLeftHandDown
  isRHandDown = isRightHandDown
  
  # If your hands up
  if not handsDirection:
    if currentLeftHandDistance < -(leftHandDistanceLevel / 8):
      isLHandDown = True
    else:
      isLHandDown = False
              
    if currentRightHandDistance < -(rightHandDistanceLevel / 8):
      isRHandDown = True
    else:
      isRHandDown = False
                
    # Changing stage due to changing direction of your hands
    
    if (not isLeftHandDown and not isRightHandDown):
      stage = "Put your hands down"
    if (isLeftHandDown and not isRightHandDown):
      stage = "Put your right hand down"
    if (not isLeftHandDown and isRightHandDown):
      stage = "Put your left hand down"
    if (isLeftHandDown and isRightHandDown):
      handsNewDirection = not handsDirection
      isRHandDown = False
      isLHandDown = False
      stage = "Put your hands up"
          
  # If your hands down
  if  handsDirection:
    if currentLeftHandDistance > (leftHandDistanceLevel / 2):
      isLHandUp = True
    else:
      isLHandUp = False
              
    if currentRightHandDistance > (rightHandDistanceLevel / 2):
      isRHandUp = True
    else:
      isRHandUp = False
      
    # Changing stage due to changing direction of your hands
                
    if (not isLeftHandUp and not isRightHandUp):
      stage = "Put your hands up"
    if (isLeftHandUp and not isRightHandUp):
      stage = "Put your right hand up"
    if (not isLeftHandUp and isRightHandUp):
      stage = "Put your left hand up"
    if (isLeftHandUp and isRightHandUp):
      handsNewDirection = not handsDirection
      counterHands = counter + 1
      isRHandUp = False
      isLHandUp = False
      stage = "Put your hands down"
  return  isLHandDown, isRHandDown , stage ,  handsNewDirection , counterHands , isRHandUp , isLHandUp
  

    
    
def countFingers(image, results, draw=True, display=True):

    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Hands initial position
    hands = 0
    leftHandFixating = 0
    rightHandFixating = 0
    leftHandPos = 0
    rightHandPos = 0
    isLeftHandMoved = False
    isRightHandMoved = False
    
    # Hands distance
    
    rightHandDistance = 0
    leftHandDistance = 0
    currentLeftHandDistance = 0
    currentrightHandDistance = 0
    
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):
        # print((results.multi_hand_landmarks[hand_index]))
        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label
        

        
        if hand_index == 0:
            firstHand = hand_info.classification[0].label
            
        if hand_index == 1 and firstHand == hand_label:
            continue 
        
        # Retrieve the landmarks of the found hand.
        hand_landmarks =  results.multi_hand_landmarks[hand_index]
        
        distanceWristMiddleFinglerMCP = (hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y)
        distanceIndexFinger =  abs(hand_landmarks.landmark[17].x - hand_landmarks.landmark[5].x)
         
        # If hand located in camera add amount of hands
        
        if (hand_landmarks.landmark[0].y < 1 and hand_landmarks.landmark[0].y > 0) \
          and (hand_landmarks.landmark[0].x < 1 and hand_landmarks.landmark[0].x > 0) \
            and (hand_landmarks.landmark[4].y < 1 and hand_landmarks.landmark[4].y > 0) \
              and (hand_landmarks.landmark[4].x < 1 and hand_landmarks.landmark[4].x > 0):
                hands += 1
                
                # If hand is close to camera add hand position
                if distanceWristMiddleFinglerMCP > 0.15 and  hand_label == "Right" and not isRightHandFixating:
                  rightHandPos = hand_landmarks.landmark[1].x
                  rightHandFixating += 1   
                  rightHandDistance = distanceWristMiddleFinglerMCP
                
                # If hand is close to camera add hand position
                
                if distanceWristMiddleFinglerMCP > 0.15 and  hand_label == "Left" and not isLeftHandFixating:
                  leftHandPos = hand_landmarks.landmark[1].x
                  leftHandFixating += 1 
                  leftHandDistance =  distanceWristMiddleFinglerMCP 
                  
        # Add hands distance
                  
        if hand_label == "Right":
          currentLeftHandDistance = distanceWristMiddleFinglerMCP   
          
        if hand_label == "Left":
          currentrightHandDistance = distanceWristMiddleFinglerMCP  
                  
        # If hands moved more than needed
        
        if hand_label == "Right" and isRightHandFixating and abs(rightHandPosition - hand_landmarks.landmark[1].x)  > 0.13:
          isRightHandMoved = True
                  
        if hand_label == "Left" and isLeftHandFixating and abs(leftHandPosition - hand_landmarks.landmark[1].x)  > 0.13:
          isLeftHandMoved = True               


              
    return output_image,  count ,  hands , rightHandFixating , leftHandFixating, rightHandPos , \
      leftHandPos, isRightHandMoved, isLeftHandMoved ,rightHandDistance , leftHandDistance, \
        currentLeftHandDistance , currentrightHandDistance
    
    
    
# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)

# Create named window for resizing purposes.
cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Perform Hands landmarks detection on the frame.
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
            # Write the total count of the fingers of both hands on the output image.
    cv2.rectangle(frame, (0, 0), (550, 73), (245, 117, 16), -1)
    cv2.putText(frame, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, 'STAGE', (115, 12),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.putText(frame, stage,
                        (115, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    if results.multi_hand_landmarks:
        
        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(frame, landmark_list = hand_landmarks,
                                      connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                   thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0),
                                                                                     thickness=2, circle_radius=2))
    
    # Check if the hands landmarks in the frame are detected.
    
        print(rightHandPosition , leftHandPosition)
            
        # Count the number of fingers up of each hand in the frame.
        frame, count , amountHands , rightHandFixating , leftHandFixating , rightHandPos, \
          leftHandPos , isRightHandMoved , isLeftHandMoved, rightHandDistance, leftHandDistance , \
            currentLeftHandDistance , currentRightHandDistance = countFingers(frame, results, display=False)
            
        
        # If rightHand is fixating add correct frames
        if (rightHandFixating == 1 and not isRightHandFixating):
          rightHandFrames += 1
        
        # If rightHand is not fixating nullify correct frames  
        if (rightHandFixating == 0 and not isRightHandFixating):
          rightHandFrames = 0
          
        # If enough correct frames just add coordinates to your hand
        if (rightHandFrames > 3 and not isRightHandFixating):
          rightHandPosition = rightHandPos
          rightHandDistanceLevel = rightHandDistance
          isRightHandFixating = not isRightHandFixating
          rightHandFrames = 0
          
        # If hand moved then add moved frames
        if (isRightHandMoved and isRightHandFixating):
          rightHandFrames += 1
          
          # If moved frames more than needed nullify your position
          if (rightHandFrames >= 6):
            rightHandPosition = 0
            isRightHandFixating = not isRightHandFixating
            rightHandFrames = 0
            
        # If right hand is not moved nullify moved frames  
        if (not isRightHandMoved and isRightHandFixating):
          rightHandFrames = 0
          
          # If leftHand is fixating add correct frames  
        if (leftHandFixating == 1 and not isLeftHandFixating):
          leftHandFrames += 1
          
        # If leftHand is not fixating nullify correct frames
        if (leftHandFixating == 0 and not isLeftHandFixating):
          leftHandFrames = 0
          
        # If enough correct frames just add coordinates to your hand
        if (leftHandFrames > 3 and not isLeftHandFixating):
          leftHandPosition = leftHandPos
          isLeftHandFixating = not isLeftHandFixating
          leftHandDistanceLevel = leftHandDistance
          leftHandFrames = 0
          
        # If hand moved then add moved frames
        if (isLeftHandMoved and isLeftHandFixating):
          leftHandFrames += 1
           # If moved frames more than needed nullify your position
           
          if (leftHandFrames >= 6):
            leftHandPosition = 0
            isLeftHandFixating = not isLeftHandFixating
            leftHandFrames = 0
            
        # If hand moved then add moved frames  
        if (not isLeftHandMoved and isLeftHandFixating):
          leftHandFrames = 0
            
        # Checking amountHands in camera
        if (amountHands == 2):
          frameCounter = 0
        else:
          frameCounter += 1
          
    
        # If hands located out of view more than certain time limit nulify your position    
        if (amountHands < 2) and frameCounter > 12:
            stage = "Too far from camera or incorrect position"
            leftHandPosition = 0
            isLeftHandFixating = False
            leftHandFrames = 0
            rightHandPosition = 0
            isRightHandFixating = False
            rightHandFrames = 0
            handsDirection = False
        else:
          # Checking hands fixating
          
          if not isLeftHandFixating:
            stage = "Your left hand isn't fixating"
            handsDirection = False
          if not isRightHandFixating:
            stage = "Your right hand isn't fixating"
            handsDirection = False
          if isRightHandFixating and isLeftHandFixating:
            isLHandDown, isRHandDown , stageHands ,  handsNewDirection , counterHands , isRHandUp , isLHandUp = handsChangeDirection(currentLeftHandDistance , currentRightHandDistance)
            isLeftHandDown = isLHandDown
            isRightHandDown = isRHandDown
            stage = stageHands
            handsDirection = handsNewDirection
            counter = counterHands
            isRightHandUp = isRHandUp
            isLeftHandUp = isLHandUp
    
    # If in this moment no hands founded nulify your position
    if (results.multi_handedness == None):
      stage = "Too far from camera or incorrect position"
      leftHandPosition = 0
      isLeftHandFixating = False
      leftHandFrames = 0
      rightHandPosition = 0
      isRightHandFixating = False
      handsDirection = False
      rightHandFrames = 0        
    # Display the frame.
    cv2.imshow('Fingers Counter', frame)
    
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()