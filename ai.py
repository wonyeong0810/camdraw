import cv2
import mediapipe as mp

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

    def is_right_hand(self, hand_index, results):
        if results.multi_handedness:
            handedness = results.multi_handedness[hand_index].classification[0].label
            return handedness == 'Right'
        return False

    def release(self):
        self.hands.close()
