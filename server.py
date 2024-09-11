from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import os

app = FastAPI()

# 비디오 파일 경로 설정
VIDEO_PATH = "recorded_output.avi"

def delete_video_file(file_path: str):
    """비디오 파일 삭제"""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} 파일이 삭제되었습니다.")

@app.get("/video")
def get_video():
    """비디오 파일을 반환한 후 파일을 삭제"""
    if os.path.exists(VIDEO_PATH):
        # 파일 반환 후 삭제 작업 추가
        return FileResponse(VIDEO_PATH, media_type='video/x-msvideo', filename="recorded_output.avi",
                            background=BackgroundTask(delete_video_file, VIDEO_PATH))
    return {"error": "Video not found."}
