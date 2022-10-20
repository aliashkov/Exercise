from google.protobuf.json_format import MessageToDict
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


counter = 0
stage = None
lastAmountFists = 0 
samePositionFrames = 0

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        max_num_hands=2,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)

        if not success:
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        currentFrameAmountFists = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                indexX = 0
                indexY = 0
                indexMid = 0
                handBottomX = 0
                handBottomY = 0
                
                for lms in lmList:
                    if lms[0] == 7:
                        indexX, indexY = lms[1], lms[2]
                    elif lms[0] == 5:
                        indexMid = lms[2]
                    elif lms[0] == 0:
                        handBottomX, handBottomY = lms[1], lms[2]

                if len(results.multi_handedness) == 2:
                    if (indexY < handBottomY) and (indexY > indexMid):
                        currentFrameAmountFists += 1     
        if currentFrameAmountFists == lastAmountFists:
            samePositionFrames += 1
        else:
            samePositionFrames = 0
        lastAmountFists = currentFrameAmountFists                
        if results.multi_handedness != None:        
          if len(results.multi_handedness) == 2:
            if currentFrameAmountFists == 0 and samePositionFrames > 0:
                if stage == 'Fists':
                   counter += 1
                stage = 'Palms'
            elif currentFrameAmountFists == 2 and samePositionFrames > 0:
                stage = 'Fists'
            elif currentFrameAmountFists == 1 and samePositionFrames > 0 and stage != 'Fists':
                stage = '1 Fist'
        cv2.rectangle(image, (0, 0), (500, 73), (245, 117, 16), -1)

        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'STAGE', (115, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if results.multi_handedness == None:
            stage = None
            cv2.putText(image, 'Put your hands on camera',
                        (115, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        elif len(results.multi_handedness) != 2:
            stage = None
            cv2.putText(image, 'Put both hands on camera',
                        (115, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, stage,
                        (115, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
