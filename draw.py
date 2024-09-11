# draw.py
import cv2
# import numpy as compy
import numpy as np

class HandPainter:
    def __init__(self, canvas_shape):
        # 캔버스 초기화 (그림을 그릴 이미지)
        self.canvas = np.zeros(canvas_shape, dtype=np.uint8)
        self.color = (255,255,255)
        self.palette = [
            (0, 0, 255),  # 빨간색
            (0, 255, 0),  # 초록색
            (255, 0, 0),  # 파란색
            (0, 255, 255),  # 노란색
            (255, 0, 255),  # 보라색
            (255, 255, 255)  # 흰색
        ]
        self.palette_rects = []
        
    
    def create_palette(self, frame):
        """캠 화면 상단에 팔레트를 표시하고, 색상 선택 기능을 구현합니다."""
        h, w, _ = frame.shape
        palette_width = 50  # 팔레트 각 색상 칸의 너비
        palette_top = 10  # 팔레트가 화면 상단에 위치하도록 설정

        for i, color in enumerate(self.palette):
            top_left = (10 + i * (palette_width + 10), palette_top)
            bottom_right = (top_left[0] + palette_width, palette_top + palette_width)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
            self.palette_rects.append((top_left, bottom_right, color))

    def check_palette_selection(self, hand_landmarks, frame):
        """왼손 검지가 팔레트 영역에 닿았는지 확인하고 색상을 선택합니다."""
        h, w, _ = frame.shape
        index_finger_tip = hand_landmarks.landmark[8]
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        for top_left, bottom_right, color in self.palette_rects:
            if top_left[0] <= cx <= bottom_right[0] and top_left[1] <= cy <= bottom_right[1]:
                self.color = color
                break

    def draw_on_canvas(self, results, frame, finger_states_dict):
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                hand_label = 'Right' if handedness == 'Right' else 'Left'
                
                # 검지 손가락이 펴져 있는지 확인
                if hand_label in finger_states_dict:
                    if hand_label == 'Left':
                        self.check_palette_selection(hand_landmarks, frame)
                    if hand_label == "Right":
                        # 검지 손가락 끝 부분의 랜드마크 (Index Finger Tip, Landmark ID: 8)
                        index_finger_tip = hand_landmarks.landmark[8]
                        h, w, c = frame.shape
                        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                        if finger_states_dict["Right"] == [1,1,0,0,0] or finger_states_dict["Right"] == [0,1,0,0,0]:
                            # 캔버스에 점 그리기
                            cv2.circle(self.canvas, (cx, cy), 5, self.color, -1)
                        if finger_states_dict["Right"] == [1,1,1,1,1]:
                            cv2.circle(self.canvas, (cx, cy), 20, (0, 0, 0), -1)

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
        # 캔버스에서 그림이 있는 부분만 추출
        
        h, w, _ = frame.shape
        canvas_h, canvas_w, _ = self.canvas.shape
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

        # 원본 프레임과 캔버스를 합성하여 그림 그린 부분만 보이게 처리
        frame_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        combined = cv2.add(frame_bg, frame_fg)

        # 현재 색상을 화면에 표시
        color_box_top_left = (w - 110, 10)
        color_box_bottom_right = (w - 10, 60)
        cv2.rectangle(frame, color_box_top_left, color_box_bottom_right, self.color, -1)

        self.create_palette(combined)

        return combined
