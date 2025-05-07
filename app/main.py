import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    DetectionResponse,
    UnprocessableEntityResponse,
)
from app.models.asset import AssetResponse, AssetType
from app.models.detection import PlasticType
from app.services.asset import get_created_asset
from app.services.gemini_service import detect_objects, request_create_asset
from app.utils.image_utils import get_origin_image, validate_image
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


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    :return: A message indicating the service is running.
    """
    return {"status": "ok"}


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
async def detect_image(image: UploadFile):
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


@app.post(
    "/assets/create",
    response_model=AssetResponse,
)
async def create_asset(model: PlasticType, asset_type: AssetType = AssetType.texture):
    """
    Create an asset in the specified type.
    :param type: The type of asset to create (e.g., textures, models).
    :param file: The file to upload.
    :return: A message indicating the success or failure of the operation.
    """

    origin_asset = get_origin_image(model=model, type=asset_type)

    try:

        response = await request_create_asset(
            image=origin_asset,
            asset_model=model.value,
            asset_type=asset_type.value,
        )
        return response

    except HTTPException as http_exc:
        # logger.debug(f"HTTP Exception: {http_exc.detail}")
        raise http_exc

    except Exception as e:
        logger.error(f"Error during asset creation: {e}")
        return AssetResponse(
            success=False,
            file=None,
        )


@app.get("/asset/", response_model=AssetResponse)
async def get_asset(model: PlasticType, asset_type: AssetType = AssetType.texture):
    """
    Get an asset in the specified type.
    :param type: The type of asset to create (e.g., textures, models).
    :param file: The file to upload.
    :return: A message indicating the success or failure of the operation.
    """

    created_asset = get_created_asset(
        model=model,
        asset_type=asset_type,
    )

    try:
        return AssetResponse(
            success=True,
            file=created_asset,
        )

    except HTTPException as http_exc:
        # logger.debug(f"HTTP Exception: {http_exc.detail}")
        raise http_exc

    except Exception as e:
        logger.error(f"Error during asset creation: {e}")
        return AssetResponse(
            success=False,
            file=None,
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
