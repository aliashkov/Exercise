import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

counter = 0
stage = None
correctHand = 0
leftHandY = 0
leftShoulderY = 0


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
            # Counter logic
            if landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x <= 0.1 or landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x >= 0.9 or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x <= 0.1 or landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x >= 0.9:
                stage = "-vis"
            elif anglePose < 165 or angleLeftShoulder < 155 or angleRightShoulder < 155:
                stage = "-pose"
            elif anglePose >= 165 and angleLeftShoulder >= 155 and angleRightShoulder >= 155 and angleLeftHand < 30 and angleRightHand > 160 and stage == '+pose' and stage != 'right':
                if wristLeftY  > (shoulderLeftY + shoulderRightY) / 2 and landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.5:
                   if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y <= ((shoulderLeftY + shoulderRightY) / 2) + 0.01  :
                      stage = "right"
                      counter += 1
                   elif landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > (shoulderLeftY + shoulderRightY) / 2 + 0.01 :
                      stage = "-hand"
            elif anglePose >= 165 and angleLeftShoulder >= 155 and angleRightShoulder >= 155 and angleLeftHand > 160 and angleRightHand < 30 and stage == '+pose' and stage != 'left':
                if wristRightY  > (shoulderLeftY + shoulderRightY) / 2 and landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.5:
                   if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y <= ((shoulderLeftY + shoulderRightY) / 2) + 0.01  :
                      stage = "left"
                      counter += 1
                   elif landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y > (shoulderLeftY + shoulderRightY) / 2 + 0.01 :
                      stage = "-hand"
            elif anglePose >= 165 and angleLeftShoulder >= 155 and angleRightShoulder >= 155 and angleLeftHand > 160 and angleRightHand < 30 and stage == 'right':
                if wristRightY  > (shoulderLeftY + shoulderRightY) / 2 and landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.5:
                   if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y <= ((shoulderLeftY + shoulderRightY) / 2) + 0.01  :
                      stage = "left"
                      counter += 1
                   elif landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y > (shoulderLeftY + shoulderRightY) / 2 + 0.01 :
                      stage = "-hand"
            elif anglePose >= 165 and angleLeftShoulder >= 155 and angleRightShoulder >= 155 and angleLeftHand < 30 and angleRightHand > 160 and stage == 'left':
                if wristLeftY  > (shoulderLeftY + shoulderRightY) / 2 and landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x <= (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 1.5:
                   if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y <= ((shoulderLeftY + shoulderRightY) / 2) + 0.01  :
                      stage = "right"
                      counter += 1
                   elif landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y > (shoulderLeftY + shoulderRightY) / 2 + 0.01 :
                      stage = "-hand"
            elif anglePose >= 165 and angleLeftShoulder >= 155 and angleRightShoulder >= 155 and angleLeftHand > 160 and angleRightHand > 160 and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x > 0.1 and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x < 0.9 and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x > 0.1 and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x < 0.9:
                stage = "+pose"

        except:
            pass

        # Setup status box
        cv2.rectangle(image, (0, 0), (320, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (115, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (115, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

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
