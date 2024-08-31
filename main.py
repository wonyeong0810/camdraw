# main.py
import cv2
from ai import HandTracker
from draw import HandPainter

def main():
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker()
    hand_painter = HandPainter(canvas_shape=(480, 640, 3))  # 캔버스 크기 설정 (예: 480p 해상도)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 상하 반전
        frame = cv2.flip(frame, 1)  # 0: 상하 반전, 1: 좌우 반전, -1: 상하 및 좌우 반전
        
        image, results = hand_tracker.process_frame(frame)
        finger_states = hand_tracker.check_fingers_extended(results)
        
        # 손의 랜드마크를 프레임에 그리기
        hand_painter.draw_landmarks(results, frame)
        
        # 검지 손가락이 펴져 있는지 체크하고 그림을 그리기
        hand_painter.draw_on_canvas(results, frame, finger_states)
        
        combined_frame = hand_painter.combine_frames(frame)
        
        # 결과를 화면에 표시
        cv2.imshow('Hand Tracking', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    hand_tracker.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
