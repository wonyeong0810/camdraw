# ai.py
import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.7):
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Mediapipe 손 인식 모듈 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
    
    def process_frame(self, frame):
        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 손의 랜드마크 탐지
        results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image, results

    def get_finger_state(self, hand_landmarks):
        # 손가락 관절 인덱스
        # 0: 엄지, 1: 검지, 2: 중지, 3: 약지, 4: 새끼손가락
        FINGER_TIP_IDS = [4, 8, 12, 16, 20]
        FINGER_PIP_IDS = [3, 7, 11, 15, 19]

        finger_states = [0] * 5
        
        for i in range(5):
            tip = hand_landmarks.landmark[FINGER_TIP_IDS[i]]
            pip = hand_landmarks.landmark[FINGER_PIP_IDS[i]]
            
            # 두 점의 거리 계산
            distance_tip_pip = np.sqrt(
                (tip.x - pip.x) ** 2 +
                (tip.y - pip.y) ** 2 +
                (tip.z - pip.z) ** 2
            )
            
            # 손가락이 펴져 있는지 판단하는 임계값
            extended_threshold = 0.040
            
            if distance_tip_pip > extended_threshold:
                finger_states[i] = 1

        return finger_states

    def check_fingers_extended(self, results):
        finger_states_dict = {}
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                hand_label = 'Right' if handedness == 'Right' else 'Left'
                
                finger_states = self.get_finger_state(hand_landmarks)
                finger_states_dict[hand_label] = finger_states
        
        return finger_states_dict

    def release(self):
        self.hands.close()
