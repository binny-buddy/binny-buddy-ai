import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models.response_models import DetectionResponse
from app.services.gemini_service import detect_objects
from app.utils.image_utils import resize_image, validate_image
from app.utils.logger import logger

app = FastAPI(title="Object Detection API with Gemini")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to Object Detection API with Gemini"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_image(image: UploadFile = None):
    try:
        # 이미지 읽기
        contents = await image.read()

        # 이미지 유효성 검사
        if not validate_image(contents):
            logger.error("Invalid image format")
            raise HTTPException(
                status_code=400, detail="유효하지 않은 이미지 형식입니다"
            )

        # 이미지 리사이징
        resized_image = resize_image(contents, max_size=(1024, 1024))

        # Gemini API로 객체 탐지 수행
        detection_result = await detect_objects(resized_image)

        return detection_result
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return DetectionResponse(success=False, objects=[], total_objects=0)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
