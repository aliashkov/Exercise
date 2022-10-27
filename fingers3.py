import cv2
import mediapipe as mp

# select which webcma to use (0 if you have just one webcam)
cap = cv2.VideoCapture(0)

# prepare detecting functions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# define finger coord
finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4,2)

while (True):
    i,image = cap.read()
    RGB_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_image)
    
    if (result.multi_hand_landmarks):
        handList=[]
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image,handLms,mp_hands.HAND_CONNECTIONS)
            for idx,lm in enumerate(handLms.landmark):
                h,w,c=image.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                handList.append((cx,cy))
        for point in handList:
            cv2.circle(image,point,10,(255,255,0),cv2.FILLED)
        upcount=0
        for coordinate in finger_Coord:
            if (handList[coordinate[0]][1] < handList[coordinate[1]][1]):
                upcount+=1
        if (handList[thumb_Coord[0]][0] > handList[thumb_Coord[1]][0]):
            upcount+=1
        cv2.putText(image,str(upcount),(150,150),cv2.FONT_HERSHEY_PLAIN,12,(0,255,0),12)
    cv2.imshow("FINGER COUNTER",image)
    # it works, until you press 'q' key on keybord
    if cv2.waitKey(1) & 0xFF == ord('q'):break
cv2.destroyAllWindows()