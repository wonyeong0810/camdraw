from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil

app = FastAPI()

# 비디오 파일 저장 경로 설정
VIDEO_PATH = "uploaded_video.avi"


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """클라이언트가 업로드한 비디오 파일을 저장"""
    try:
        with open(VIDEO_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "Video uploaded successfully!"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        file.file.close()


@app.get("/video")
def get_video():
    """저장된 비디오 파일을 제공"""
    if os.path.exists(VIDEO_PATH):
        return FileResponse(VIDEO_PATH, media_type='video/x-msvideo', filename="recorded_output.avi")
    return {"error": "Video not found."}


@app.delete("/delete/")
def delete_video():
    """비디오 파일을 삭제"""
    try:
        if os.path.exists(VIDEO_PATH):
            os.remove(VIDEO_PATH)
            return {"message": "Video deleted successfully!"}
        else:
            return {"error": "Video file not found."}
    except Exception as e:
        return {"error": str(e)}
