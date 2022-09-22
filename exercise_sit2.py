import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

counter = 0
stage = None
lastStage = None
leftShoulderCheck = False
rightShoulderCheck = False
leftShoulderCounter = 0
rightShoulderCounter = 0
rightWristCounter = 0
leftWristCounter = 0
rightWristX = None
leftWristX = None
rightWristXCounter = 0
leftWristXCounter = 0
shouldersLevelCounter = 0

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def calculate_angle_2(a, b):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    radians = np.arctan2(b[1]-a[1], b[0]-a[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detection
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates Left
            shoulderLeft = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbowLeft = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wristLeft = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get coordinates Right
            shoulderRight = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbowRight = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wristRight = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            anglePose = calculate_angle_2(shoulderLeft, shoulderRight)
            angleLeftShoulder = calculate_angle_2(elbowLeft, shoulderLeft)
            angleRightShoulder = calculate_angle_2(shoulderRight, elbowRight)

            angleLeftHand = calculate_angle(shoulderLeft, elbowLeft, wristLeft)
            angleRightHand = calculate_angle(
                shoulderRight, elbowRight, wristRight)

            wristLeftY = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            wristRightY = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            shoulderLeftY = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            shoulderRightY = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

            # Visualize angle
            cv2.putText(image, str(angleLeftHand),
                        tuple(np.multiply(elbowLeft, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                        255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(angleRightHand),
                        tuple(np.multiply(elbowRight, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,
                                                        255, 255), 2, cv2.LINE_AA
                        )

            # Left Shoulder on right position
            
            if (leftShoulderCheck == False and angleLeftShoulder > 140):
                leftShoulderCounter += 1
                
            # Fixating position after certain frames
            if (leftShoulderCheck == False and leftShoulderCounter >= 1):
                if  angleLeftShoulder <= 140 :
                    leftShoulderCounter = 0
                if leftShoulderCounter >= 12:
                    leftShoulderCheck = True
            
            # Left Shoulder on wrong position
                    
            if (leftShoulderCheck == True and angleLeftShoulder <= 140):
                leftShoulderCounter += 1
                
            # Fixating position after certain frames
                
            if (leftShoulderCheck == True and leftShoulderCounter >= 1):
                if  angleLeftShoulder > 140 :
                    leftShoulderCounter = 0
                if leftShoulderCounter >= 12:
                    leftShoulderCheck = False
                    
                    
                    
                    
            # Right Shoulder on right position        
                        
            if (rightShoulderCheck == False and angleRightShoulder > 140):
                rightShoulderCounter += 1
                
            # Fixating position after certain frames
                
            if (rightShoulderCheck == False and rightShoulderCounter >= 1):
                if  angleRightShoulder <= 140 :
                    rightShoulderCounter = 0
                if rightShoulderCounter >= 12:
                    rightShoulderCheck = True
                    
            # Right Shoulder on wrong position   
                    
            if (rightShoulderCheck == True and angleRightShoulder <= 140):
                rightShoulderCounter += 1
                
             # Fixating position after certain frames
                
            if (rightShoulderCheck == True and rightShoulderCounter >= 1):
                if  angleRightShoulder > 140 :
                    rightShoulderCounter = 0
                if rightShoulderCounter >= 12:
                    rightShoulderCheck = False
                    
            
            # Fixating right wrist after certain frames
                    
            if (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x <= 0 or landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x >= 1):
                rightWristCounter = 0
            elif (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x > 0 and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x < 1):
                rightWristCounter += 1     
            if (rightWristCounter < 4):
                angleRightHand = 180
                
                
            # Fixating left wrist after certain frames
                
            if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x <= 0 or landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x >= 1):
                leftWristCounter = 0
            elif (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x > 0 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x < 1):
                leftWristCounter += 1
            if leftWristCounter < 4:
                angleLeftHand = 180
            
            # Disable right wrist angle if positions varies too different
                
            if (rightWristX != None):
               if rightWristX - 0.05 <= landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x  >= rightWristX - 0.05:
                   rightWristXCounter += 1
                   rightWristX = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
               elif rightWristX - 0.05 > landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x or landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x  < rightWristX - 0.05:
                   rightWristXCounter = 0
                   rightWristX = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
            elif rightWristX == None:
               rightWristX = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
               
               
            # Disable left wrist angle if positions varies too different
               
            if (leftWristX != None):
               if leftWristX - 0.05 <= landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x  >= leftWristX - 0.05:
                   leftWristXCounter += 1
                   leftWristX = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
               elif leftWristX - 0.05 > landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x or landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x  < leftWristX - 0.05:
                   leftWristXCounter = 0
                   leftWristX = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
            elif leftWristX == None:
               leftWristX = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
                
            

            
            # Exercise 
            
            if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x > 0.75 or landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y > 0.7 or landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x < 0.3 or landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y > 0.7:
                stage = "you are < 60sm or you sit far from center"
            elif anglePose < 160 or leftShoulderCheck == False or rightShoulderCheck == False and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x <= 0.7 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= 0.7 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x >= 0.3 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y <= 0.7:
                if lastStage == None:
                    stage = "return hands to the sides"
                elif lastStage == "right position":
                    stage = "return right hand to the side, left hand to the elbow"
                elif lastStage == "left position":
                    stage = "return left hand to the side, right hand to the elbow"
            elif anglePose >= 160 and leftShoulderCheck == True and rightShoulderCheck == True and angleLeftHand < 30 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x / 2.5 <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)  and (angleRightHand > 160) and stage == 'initial pose' and stage != 'right position':
                if wristLeftY > ((shoulderLeftY + shoulderRightY) / 2) - 0.07 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x / 2.5 <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) :
                    if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > shoulderLeftY + 0.05 and leftWristXCounter > 4:
                        lastStage = None
                        stage = "return hands to the sides"
                    elif landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y <= shoulderLeftY + 0.05  and leftWristXCounter > 1  and stage != 'return hands to the sides':
                        stage = "right position"
                        lastStage = "right position"
                        counter += 1
            elif anglePose >= 160 and leftShoulderCheck == True  and rightShoulderCheck == True and angleLeftHand > 160 and angleRightHand < 30  and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.8 and stage == 'initial pose' and stage != 'left position':
                if wristRightY > ((shoulderLeftY + shoulderRightY) / 2) - 0.07 and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x   <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.8 :
                    if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y > shoulderRightY + 0.05 and rightWristXCounter > 4:
                        lastStage = None
                        stage = "return hands to the sides"
                    elif landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y <= shoulderRightY + 0.05 and rightWristXCounter > 1 and stage != 'return hands to the sides':
                        stage = "left position"
                        lastStage = "left position"
                        counter += 1

            elif anglePose >= 160 and leftShoulderCheck == True and rightShoulderCheck == True and angleRightHand > 120 and stage == 'right position':
                angleLeftHand1 = calculate_angle(shoulderLeft, elbowLeft, wristLeft)
                angleRightHand1 = calculate_angle(shoulderRight, elbowRight, wristRight)
                if (angleLeftHand1 + angleRightHand1 > 280):
                    shouldersLevelCounter += 1
                elif (angleLeftHand1 + angleRightHand1 <= 280):
                    shouldersLevelCounter = 0
                if (shouldersLevelCounter > 12):
                   stage = "return right hand to the side, left hand to the elbow"
                   lastStage = "right position"
            elif anglePose >= 160 and leftShoulderCheck == True  and rightShoulderCheck == True and angleLeftHand > 160  and angleRightHand < 30  and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.8 and stage == 'right position':

                if  wristRightY > ((shoulderLeftY + shoulderRightY) / 2) - 0.07 and landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.8:
                    if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y > shoulderRightY + 0.05 and rightWristXCounter > 4:
                        stage = "return right hand to the side, left hand to the elbow"
                        lastStage = "right position"
                    elif landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y <= shoulderRightY + 0.05 and rightWristXCounter > 1 and stage != 'return right hand to the side, left hand to the elbow':
                        stage = "left position"
                        lastStage = "left position"
                        counter += 1
            
            elif anglePose >= 160 and leftShoulderCheck == True and rightShoulderCheck == True and angleLeftHand > 120 and stage == 'left position':
                angleLeftHand1 = calculate_angle(shoulderLeft, elbowLeft, wristLeft)
                angleRightHand1 = calculate_angle(shoulderRight, elbowRight, wristRight)
                if (angleLeftHand1 + angleRightHand1 > 280):
                    shouldersLevelCounter += 1
                elif (angleLeftHand1 + angleRightHand1 <= 280):
                    shouldersLevelCounter = 0
                if (shouldersLevelCounter > 12):
                   stage = "return left hand to the side, right hand to the elbow"
                   lastStage = "left position"
            elif anglePose >= 160 and leftShoulderCheck == True and rightShoulderCheck == True and angleLeftHand < 30 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x / 2.5 <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)  and (angleRightHand > 160) and stage == 'left position':
                
                if  wristLeftY > ((shoulderLeftY + shoulderRightY) / 2) - 0.07 and landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x / 2.5 <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) :
                    if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > shoulderLeftY + 0.05 and leftWristXCounter > 4:
                        stage = "return left hand to the side, right hand to the elbow"
                        lastStage = "left position"
                    elif landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y <= shoulderLeftY + 0.05 and leftWristXCounter > 1 and stage != 'return left hand to the side, right hand to the elbow':
                        stage = "right position"
                        lastStage = "right position"
                        counter += 1
            elif lastStage == None and anglePose >= 160 and leftShoulderCheck == True and rightShoulderCheck == True and angleLeftHand > 160 and angleRightHand > 160 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x <= 0.7 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x >= 0.3 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= 0.7 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y <= 0.7:
                stage = "initial pose"
            elif lastStage == "right position" and anglePose >= 160 and leftShoulderCheck == True and rightShoulderCheck == True and angleLeftHand < 30 and angleRightHand > 160 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x <= 0.7 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x >= 0.3 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= 0.7 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y <= 0.7:
                stage = "right position"
            elif lastStage == "left position" and anglePose >= 160 and leftShoulderCheck == True and rightShoulderCheck == True and angleLeftHand > 160 and angleRightHand < 30 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x <= 0.7 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x >= 0.3 and landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= 0.7 and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y <= 0.7:
                stage = "left position"

        except:
            pass


        # Setup status box
        cv2.rectangle(image, (0, 0), (600, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (115, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if len(stage) or stage == 'None' < 15:
          cv2.putText(image, stage,
                    (115, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        elif len(stage) >= 15 and len(stage) < 25:
          cv2.putText(image, stage,
                    (115, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        elif len(stage) >= 25:
          cv2.putText(image, stage,
                    (115, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 2, cv2.LINE_AA)
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()