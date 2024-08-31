# draw.py
import cv2
import numpy as np

class HandPainter:
    def __init__(self, canvas_shape):
        # 캔버스 초기화 (그림을 그릴 이미지)
        self.canvas = np.zeros(canvas_shape, dtype=np.uint8)

    def draw_on_canvas(self, results, frame, finger_states_dict):
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                hand_label = 'Right' if handedness == 'Right' else 'Left'
                
                # 검지 손가락이 펴져 있는지 확인
                if hand_label in finger_states_dict:
                    if finger_states_dict[hand_label] == [1,1,0,0,0] or finger_states_dict[hand_label] == [0,1,0,0,0]:
                        # 검지 손가락 끝 부분의 랜드마크 (Index Finger Tip, Landmark ID: 8)
                        index_finger_tip = hand_landmarks.landmark[8]
                        h, w, c = frame.shape
                        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                        # 캔버스에 점 그리기
                        cv2.circle(self.canvas, (cx, cy), 5, (255, 255, 255), -1)

    def draw_landmarks(self, results, frame):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    # 손가락 랜드마크 위치
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # 랜드마크 그리기
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # 손가락 관절을 선으로 연결
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
                ]
                for start, end in connections:
                    start_landmark = hand_landmarks.landmark[start]
                    end_landmark = hand_landmarks.landmark[end]
                    start_x, start_y = int(start_landmark.x * w), int(start_landmark.y * h)
                    end_x, end_y = int(end_landmark.x * w), int(end_landmark.y * h)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    def clear_canvas(self):
        # 캔버스를 지우기 (검은색으로 초기화)
        self.canvas = np.zeros(self.canvas.shape, dtype=np.uint8)
    
    def combine_frames(self, frame):
        # 원본 프레임과 캔버스를 결합하여 출력
        return cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
