import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, model_complexity=1)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5 ,model_complexity=1)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

counter = 0
fingerCount = 0

reverseDirection = False
stage = "Too far from camera or incorrect position"
    
    
def countFingers(image, results, draw=True, display=True):

    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    
    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    
    hands = 0
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
        
        distanceWristMiddleFinglerMCP = abs(hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y)
        distanceIndexFinger =  abs(hand_landmarks.landmark[17].x - hand_landmarks.landmark[5].x)
    
    
        
        if (hand_landmarks.landmark[0].y < 1 and hand_landmarks.landmark[0].y > 0) \
          and (hand_landmarks.landmark[0].x < 1 and hand_landmarks.landmark[0].x > 0) \
            and (hand_landmarks.landmark[4].y < 1 and hand_landmarks.landmark[4].y > 0) \
              and (hand_landmarks.landmark[4].x < 1 and hand_landmarks.landmark[4].x > 0) and distanceWristMiddleFinglerMCP > 0.15:
                if (hand_label == "Left" and hand_landmarks.landmark[1].x > hand_landmarks.landmark[0].x \
                  and hand_landmarks.landmark[17].x  < hand_landmarks.landmark[1].x \
                    and distanceWristMiddleFinglerMCP / 6 < distanceIndexFinger) or \
                      (hand_label == "Right" and hand_landmarks.landmark[1].x  <  hand_landmarks.landmark[0].x \
                        and hand_landmarks.landmark[17].x  > hand_landmarks.landmark[1].x \
                          and distanceWristMiddleFinglerMCP / 6 < distanceIndexFinger):
                            hands += 1
                            

        
        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:
            
            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]
            
            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                
                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1
        
        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
        
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
     
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
     
    thumbs_down = (not fingers_statuses["LEFT_THUMB"]) and (not fingers_statuses['RIGHT_THUMB'])
    indexes_down =  (not fingers_statuses["LEFT_INDEX"]) and (not fingers_statuses['RIGHT_INDEX'])
    middles_down = (not fingers_statuses["LEFT_MIDDLE"]) and (not fingers_statuses['RIGHT_MIDDLE'])
    ringes_down = (not fingers_statuses["LEFT_RING"]) and (not fingers_statuses['RIGHT_RING'])
    pinkies_down = (not fingers_statuses["LEFT_PINKY"]) and (not fingers_statuses['RIGHT_PINKY'])
    thumbs_up = (fingers_statuses["LEFT_THUMB"]) and (fingers_statuses['RIGHT_THUMB'])
    indexes_up =  (fingers_statuses["LEFT_INDEX"]) and (fingers_statuses['RIGHT_INDEX'])
    middles_up = (fingers_statuses["LEFT_MIDDLE"]) and (fingers_statuses['RIGHT_MIDDLE'])
    ringes_up = (fingers_statuses["LEFT_RING"]) and (fingers_statuses['RIGHT_RING'])
    pinkies_up = (fingers_statuses["LEFT_PINKY"]) and (fingers_statuses['RIGHT_PINKY'])

    fingCount = fingerCount  
    

    if (hands == 2):
        if (not reverseDirection):
          match fingerCount:
            case 0:
                if thumbs_down:
                   fingCount += 1
            case 1:
                if thumbs_down and indexes_down:
                   fingCount += 1
            case 2:
                if thumbs_down and indexes_down and middles_down:
                   fingCount += 1
            case 3:
                if (thumbs_down and indexes_down and middles_down \
                   and ringes_down):
                       fingCount += 1
            case 4:
                if (thumbs_down and indexes_down and middles_down \
                   and ringes_down and pinkies_down):
                       fingCount += 1
        else:
          match fingerCount:
            case 1:
                if (thumbs_up and indexes_up and middles_up \
                    and ringes_up and pinkies_up):
                       fingCount -= 1
            case 2:
                if (indexes_up and middles_up \
                    and ringes_up and pinkies_up):
                       fingCount -= 1
            case 3:
                if (indexes_down and middles_up \
                    and ringes_up and pinkies_up):
                       fingCount -= 1
            case 4:
                if (indexes_down and middles_down \
                    and ringes_up and pinkies_up):
                       fingCount -= 1
            case 5:
                if (indexes_down and middles_down \
                   and ringes_down and pinkies_up):
                       fingCount -= 1
         
    
    return output_image, fingers_statuses, count , fingCount, hands
    
    
    
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
    if results.multi_hand_landmarks:
            
        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count , fingCount , amountHands = countFingers(frame, results, display=False)
        

        
        if fingCount != fingerCount:
            fingerCount = fingCount
        
        if (fingerCount == 5 and not reverseDirection) or ((fingerCount == 0 and reverseDirection)):
            if (fingerCount == 0 and reverseDirection):
                counter += 1
            reverseDirection = not reverseDirection 
            
        if (amountHands < 2):
            stage = "Too far from camera or incorrect position"
        else:
          if (not reverseDirection):
            match fingerCount:
              case 0:
                stage = "Thumbs down"
              case 1:
                stage = "Indexes down"  
              case 2:
                stage = "Middles down"    
              case 3:
                stage = "Ringes down"   
              case 4:
                stage = "Pinkies down"     
          else:
            match fingerCount:  
              case 1:
                stage = "Thumbs up"
              case 2:
                stage = "Indexes up"  
              case 3:
                stage = "Middles up"    
              case 4:
                stage = "Ringes up"   
              case 5:
                stage = "Pinkies up"   
          
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