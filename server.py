from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI()

# 비디오 파일 경로 설정
VIDEO_PATH = "recorded_output.avi"

@app.get("/video")
def get_video():
    """비디오 파일을 반환"""
    if os.path.exists(VIDEO_PATH):
        return FileResponse(VIDEO_PATH, media_type='video/x-msvideo', filename="recorded_output.avi")
    return {"error": "Video not found."}
