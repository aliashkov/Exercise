import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

counter = 0
stage = None
lastStage = None
handsLevelCounter = 0

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


def calculate_angle_level(angle_counter , first_angle , second_angle):
    
    if (first_angle + second_angle >= 25)  and (first_angle + second_angle < 320):
        angle_counter = 0
    elif (first_angle + second_angle < 25)  or (first_angle + second_angle > 320):
        angle_counter += 1  
    return angle_counter
       


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

            angleLeftHand = calculate_angle(shoulderLeft, elbowLeft, wristLeft)
            angleRightHand = calculate_angle(
                shoulderRight, elbowRight, wristRight)

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
            
            #Checking angles after certain frames
        
            angleCounter = calculate_angle_level(
                handsLevelCounter, angleLeftHand, angleRightHand)
            
            handsLevelCounter = angleCounter
           
            # Counter logic
            print(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x)
            
            if (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x <= 0.01 or landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x >= 0.99) \
                or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x <= 0.01 or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x >= 0.99 \
                  or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y >= 1.05 or landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y >= 1.05 :
                    stage = "out of vision"
            elif stage == "out of vision":
                if lastStage == None:
                    stage = "return hands to the initial pose"
                elif lastStage == "right position":
                    stage = "return right hand to the side, left hand to the elbow"
                elif lastStage == "left position":
                    stage = "return left hand to the side, right hand to the elbow"
            elif angleLeftHand + angleRightHand > 300 and stage == 'initial pose':
                    lastStage = None
                    stage = "return hands to the initial pose"   
            elif angleLeftHand < 15 and angleRightHand > 80 and stage == 'initial pose'  and stage != 'right position':
                if (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y <= landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y - 0.1) \
                    or (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x > landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x \
                        or landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - 0.3):
                           lastStage = None
                           stage = "return hands to the initial pose"
                elif landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y > landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y - 0.1 \
                    and stage != 'return hands to the initial pose':
                       stage = "right position"
                       lastStage = "right position"
                       counter += 1
            elif angleLeftHand > 80 and angleRightHand < 15 and stage == 'initial pose' and stage != 'left position':
                if (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y <= landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y - 0.1) \
                    or (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x < landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x \
                        or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x + 0.3):
                           lastStage = None
                           stage = "return hands to the initial pose"
                elif landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y - 0.1 \
                    and  stage != 'return hands to the initial pose':
                    stage = "left position"
                    lastStage = "left position"
                    counter += 1
            elif  (angleLeftHand + angleRightHand > 300  or  handsLevelCounter > 8)   and stage == 'right position':
                stage = "return right hand to the side, left hand to the elbow"
                lastStage = "right position"
            elif stage == 'right position'  \
                and  ( landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x  > landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x \
                    or landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 0.3 ):
                       stage = "return right hand to the side, left hand to the elbow"
                       lastStage = "right position" 
            elif angleLeftHand > 80 and angleRightHand < 15 and stage == 'right position':
                if  (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y <= landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y - 0.1) \
                    or  (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x  > landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x \
                        or landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 0.3 ):
                                stage = "return right hand to the side, left hand to the elbow"
                                lastStage = "right position"
                elif (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y - 0.1) \
                    and  stage != 'return right hand to the side, left hand to the elbow':
                       stage = "left position"
                       lastStage = "left position"
                       counter += 1
            elif  (angleLeftHand + angleRightHand > 300  or  handsLevelCounter > 8)  and stage == 'left position':
                stage = "return left hand to the side, right hand to the elbow"
                lastStage = "left position"
            elif stage == 'left position' \
                and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x < landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x  \
                    or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x - 0.3):
                           stage = "return left hand to the side, right hand to the elbow"
                           lastStage = "left position" 
            elif  angleLeftHand < 15 and angleRightHand > 80 and stage == 'left position':
                if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y <= landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y - 0.1 \
                   and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x < landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x  \
                        or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x - 0.3):
                            stage = "return left hand to the side, right hand to the elbow"
                            lastStage = "left position"
                elif (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y > landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y - 0.1) \
                    and stage != 'return left hand to the side, right hand to the elbow':
                       stage = "right position"
                       lastStage = "right position"
                       counter += 1
            elif lastStage == None and angleLeftHand < 15 and angleRightHand < 15 and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x > 0.01 \
                and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x < 0.99 and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x > 0.01 \
                    and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x < 0.99 and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y < 1.05 \
                        and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y < 1.05 \
                            and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x >= landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x \
                                and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x >= landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x - 0.3 \
                                    and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x <= landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x \
                                        and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x <= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 0.3 ) :
                                           stage = "initial pose"
            elif lastStage == "right position" and  angleLeftHand < 15 and angleRightHand > 80 \
                and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x > 0.01 and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x < 0.99 \
                    and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x > 0.01 and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x < 0.99 \
                        and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y < 1.05 and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y < 1.05 \
                            and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x >= landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x \
                                and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x >= landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x - 0.3 \
                                    and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x <= landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x \
                                        and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x <= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 0.3 ) :
                                           stage = "right position"
            elif lastStage == "left position" and   angleLeftHand > 80 and angleRightHand < 15 \
                and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x > 0.01 and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x < 0.99 \
                    and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x > 0.01 and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x < 0.99 \
                        and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y < 1.05 and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y < 1.05 \
                            and (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x >= landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x \
                                and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x >= landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x - 0.3 \
                                    and landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x <= landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x \
                                        and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x <= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + 0.3 ) :
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
        if len(stage) < 15:
            cv2.putText(image, stage,
                    (115, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        elif len(stage) >= 15 and len(stage) < 25:
            cv2.putText(image, stage,
                    (115, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        if len(stage) >= 25:
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
