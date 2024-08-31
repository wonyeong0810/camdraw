import cv2
import numpy as np

class HandPainter:
    def __init__(self, canvas_shape):
        # 캔버스 초기화 (그림을 그릴 이미지)
        self.canvas = np.zeros(canvas_shape, dtype=np.uint8)

    def draw_on_canvas(self, results, frame, is_right_hand):
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if is_right_hand(i, results):  # 오른손인지 확인
                    # 검지 손가락 끝 부분의 랜드마크 (Index Finger Tip, Landmark ID: 8)
                    index_finger_tip = hand_landmarks.landmark[8]
                    h, w, c = frame.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                    # 캔버스에 점 그리기
                    cv2.circle(self.canvas, (cx, cy), 5, (255, 255, 255), -1)

    def combine_frames(self, frame):
        # 원본 프레임과 캔버스를 결합하여 출력
        return cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
