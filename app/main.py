import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models.response_models import DetectionResponse, UnprocessableEntityResponse
from app.services.gemini_service import detect_objects
from app.utils.image_utils import validate_image
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


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        422: {
            "model": UnprocessableEntityResponse,
            "description": "Invalid image format",
        }
    },
)
async def detect_image(image: UploadFile = None):
    try:
        if image is None:
            logger.info("No image provided")
            raise HTTPException(status_code=422, detail="No image provided")

        contents = await image.read()

        # 이미지 유효성 검사
        if not validate_image(contents):
            logger.info("Invalid image format")
            raise HTTPException(status_code=422, detail="Invalid image format")

        # Gemini API로 객체 탐지 수행
        detection_result = await detect_objects(contents)

        return detection_result
    except HTTPException as http_exc:
        # logger.debug(f"HTTP Exception: {http_exc.detail}")
        raise http_exc

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return DetectionResponse(success=False, objects=[], total_objects=0)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
