import cv2
import mediapipe as mp
import numpy as np
import math


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def euclidean_distance(point1, point2):
    return math.dist([point1.x, point1.y], [point2.x, point2.y])


def is_triangle(p1, p2, p3, threshold=0.01):
    area = abs((p1.x * (p2.y - p3.y) +
                p2.x * (p3.y - p1.y) +
                p3.x * (p1.y - p2.y)) / 2.0)
    return area > threshold


def finger_states(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]
    finger_pip = [6, 10, 14, 18]
    results = []


    if hand_label == "Right":
        results.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        results.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    
    for tip, pip in zip(finger_tips, finger_pip):
        results.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)

    return results  


def classify_gesture(hands_data):
    
    if len(hands_data) == 2:
        f1, lm1, label1 = hands_data[0]
        f2, lm2, label2 = hands_data[1]

        
        avg_x1 = np.mean([p.x for p in lm1.landmark])
        avg_x2 = np.mean([p.x for p in lm2.landmark])

        
        if (label1 == "Left" and avg_x1 > avg_x2) or (label1 == "Right" and avg_x1 < avg_x2):
            return "DANGER :("

        
        palm_dist = euclidean_distance(lm1.landmark[0], lm2.landmark[0])
        if palm_dist < 0.1 and f1 == [0,1,1,1,0] and f2 == [0,1,1,1,0]:
            return "PLEASE HELP"

        
        if f1 == [1,1,1,1,1] and f2 == [1,1,1,1,1]:
            return "HELLO !!"

    
    if len(hands_data) == 1:
        f, lm, label = hands_data[0]

        # 🖐️ HELLO
        if f == [1,1,1,1,1]:
            return "HELLO !!"

        # 👊 Fist
        elif f == [0,0,0,0,0]:
            return "YES"

        # ☝️ One finger up
        elif f == [0,1,0,0,0]:
            return "NO (X)"

        # 🤟 I LOVE YOU
        elif f == [1,1,0,0,1]:
            return "I LOVE YOU <3"

        # ✌️ Victory
        elif f == [0,1,1,0,0]:
            return "THANK YOU SO MUCH :)"

        # 👍 YES
        elif f == [1,0,0,0,0]:
            return "PLEASE HELP"

        # 👌 OK → thumb tip close to index tip, other fingers extended
        thumb_tip = lm.landmark[4]
        index_tip = lm.landmark[8]
        dist = euclidean_distance(thumb_tip, index_tip)
        if dist < 0.05 and f[2] == 1 and f[3] == 1 and f[4] == 1:
            return "OK"

        # 🤙 CALL ME → thumb & pinky up, rest down (palm or back)
        if f == [1,0,0,0,1]:
            # Use z-axis (depth) difference between wrist & middle MCP
            wrist_z = lm.landmark[0].z
            middle_mcp_z = lm.landmark[9].z

            # Palm facing camera OR back facing camera → accept both
            if abs(wrist_z - middle_mcp_z) < 0.15 or wrist_z < middle_mcp_z or wrist_z > middle_mcp_z:
                return "CALL ME LATER"

    return None




cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = None
    hand_data = []

    if result.multi_hand_landmarks and result.multi_handedness:
        for lm, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            fingers = finger_states(lm, handedness.classification[0].label)
            hand_data.append((fingers, lm, handedness.classification[0].label))

        
        label = classify_gesture(hand_data)

        
        if label:
            
            all_x, all_y = [], []
            for _, lm, _ in hand_data:
                all_x.extend([landmark.x for landmark in lm.landmark])
                all_y.extend([landmark.y for landmark in lm.landmark])

            xmin = int(min(all_x) * w) - 20
            xmax = int(max(all_x) * w) + 20
            ymin = int(min(all_y) * h) - 20
            ymax = int(max(all_y) * h) + 20

            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Sign Language Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
