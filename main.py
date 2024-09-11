import cv2
import qrcode
from ai import HandTracker
from draw import HandPainter
import os
import numpy as np

# FastAPI 서버 주소 설정 (로컬 서버의 IP 주소나 도메인으로 대체)
SERVER_URL = "http://127.0.0.1:8000/video"

def main():
    # 비디오 파일 저장 설정
    output_path = 'recorded_output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(0)
    
    # 화면 크기 가져오기
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # 비디오 저장 객체 생성
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    hand_tracker = HandTracker()
    hand_painter = HandPainter(canvas_shape=(height, width, 3))  # 캔버스 크기 설정

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # 0: 상하 반전, 1: 좌우 반전, -1: 상하 및 좌우 반전
        
        image, results = hand_tracker.process_frame(frame)
        finger_states = hand_tracker.check_fingers_extended(results)
        
        # 손의 랜드마크를 프레임에 그리기
        hand_painter.draw_landmarks(results, frame)
        
        # 양손의 모든 손가락이 펴져 있는지 확인
        all_fingers_extended = (
            len(finger_states) == 2  # 양손 모두 인식된 경우
            and all(all(state == 1 for state in states) for states in finger_states.values())
        )
        
        if all_fingers_extended:
            hand_painter.clear_canvas()
            
        # 검지 손가락이 펴져 있는지 체크하고 그림을 그리기
        hand_painter.draw_on_canvas(results, frame, finger_states)
        
        combined_frame = hand_painter.combine_frames(frame)

        # 비디오에 현재 프레임 기록
        out.write(combined_frame)

        # 결과를 화면에 표시
        cv2.imshow('Hand Tracking', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 비디오 녹화 종료
    cap.release()
    out.release()
    hand_tracker.release()
    cv2.destroyAllWindows()

    # QR 코드 생성 및 표시
    show_qr_code()

def show_qr_code():
    # FastAPI 서버의 비디오 URL을 QR 코드로 변환
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    # 비디오 파일이 제공될 URL
    qr.add_data(SERVER_URL)
    qr.make(fit=True)

    # QR 코드 이미지를 생성
    img = qr.make_image(fill='black', back_color='white')

    # OpenCV로 QR 코드 표시
    qr_image = np.array(img.convert('RGB'))
    cv2.imshow('QR Code', qr_image)
    
    # 'q' 키를 누르면 QR 코드 창을 닫음
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
