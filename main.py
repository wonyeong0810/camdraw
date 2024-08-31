import cv2
from ai import HandTracker
from draw import HandPainter

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return
    
    # HandTracker와 HandPainter 초기화
    tracker = HandTracker()
    painter = HandPainter(canvas_shape=frame.shape)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 좌우 반전
        frame = cv2.flip(frame, 1)

        # 프레임 처리 및 랜드마크 탐지
        frame, results = tracker.process_frame(frame)
        
        # 오른손으로 그림 그리기
        painter.draw_on_canvas(results, frame, tracker.is_right_hand)
        
        # 프레임과 캔버스를 결합하여 출력
        output_frame = painter.combine_frames(frame)
        
        # 결과 이미지 출력
        cv2.imshow('Hand Painting', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()

if __name__ == "__main__":
    main()
