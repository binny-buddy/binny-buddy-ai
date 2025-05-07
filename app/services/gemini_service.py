import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from PIL import Image
from pydantic import TypeAdapter

from app.models import (
    AssetFile,
    AssetResponse,
    DetectedObject,
    DetectionResponse,
    PlasticType,
    WasteStatus,
)
from app.utils.logger import logger

# 환경 변수 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# Gemini 클라이언트 설정
client = genai.Client(api_key=GEMINI_API_KEY)


CONFIDENCE_THRESHOLD = 0.6  # 최소 신뢰도 설정


async def call_gemini_api(image: Image.Image) -> GenerateContentResponse:
    """
    Gemini API를 호출하여 이미지 분석 결과를 얻습니다.

    Args:
        image: PIL Image 객체

    Returns:
        Any: Gemini API 응답
    """
    # 프롬프트 준비
    text_input = (
        "Detect all plastic waste in the image. "
        "Label will be one of the following: "
        f"{', '.join([label.value for label in PlasticType])}. "
        "The box_2d should be [ymin, xmin, ymax, xmax] "
        f"Describe its status as either {WasteStatus.clean} or {WasteStatus.dirty}. "
        "Provide a how_to_recycle description for each detected object. "
    )

    return client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[text_input, image],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[DetectedObject],
        },
    )


def parse_gemini_response(response: GenerateContentResponse) -> List[DetectedObject]:
    detected_objects = []

    if not response.candidates or not response.candidates[0].content.parts:
        logger.info("Gemini API 응답이 비어있습니다.")
        return detected_objects

    try:
        response_text = response.candidates[0].content.parts[0].text

        json_match = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
        if json_match:
            objects_list = json.loads(json_match.group())

            # TypeAdapter 사용
            adapter = TypeAdapter(List[DetectedObject])
            parsed_objects = adapter.validate_python(objects_list)

            detected_objects = [
                obj for obj in parsed_objects if obj.confidence >= CONFIDENCE_THRESHOLD
            ]
        else:
            logger.debug("응답에서 JSON 형식의 객체를 찾을 수 없습니다.")

    except Exception as e:
        logger.debug(f"응답 파싱 오류: {e}")

    return detected_objects


def create_detection_response(
    success: bool, objects: List[DetectedObject], error_message: Optional[str] = None
) -> DetectionResponse:
    """
    탐지 결과 응답을 생성합니다.

    Args:
        success: 성공 여부
        objects: 탐지된 객체 목록
        error_message: 오류 메시지 (실패 시)

    Returns:
        DetectionResponse: 탐지 결과 응답
    """
    if success:
        return DetectionResponse(
            success=True, objects=objects, total_objects=len(objects)
        )
    else:
        logger.debug(f"탐지 실패: {error_message}")
        return DetectionResponse(success=False, objects=[], total_objects=0)


async def detect_objects(image_data: bytes) -> DetectionResponse:
    """
    Gemini API를 사용하여 이미지에서 객체를 탐지하는 함수

    Args:
        image_data: 이미지 바이너리 데이터

    Returns:
        DetectionResponse: 탐지된 객체 정보와 분석 결과
    """
    try:
        # 이미지 로드
        image = Image.open(BytesIO(image_data))

        # Gemini API 호출 (schema 지정하지 않음)
        response = await call_gemini_api(image)

        # 응답 파싱
        detected_objects = parse_gemini_response(response)

        # 결과 응답 생성
        if not detected_objects:
            return create_detection_response(False, [], "No plastic waste detected.")

        # 결과 응답 생성
        return create_detection_response(True, detected_objects)

    except Exception as e:
        logger.debug(f"탐지 중 오류 발생: {e}")
        return create_detection_response(False, [], str(e))


async def request_create_asset(
    image: Image.Image,
    asset_model: str,
    asset_type: str,
) -> AssetResponse:
    """
    Gemini API를 사용하여 자산을 생성하는 함수
    """

    try:
        # 프롬프트 준비
        text_input = (
            "This is a texture for a 3D model. "
            "Please generate cute variations of this texture "
            "Do not change the shape or structure of the texture. "
            "Only modify the center red section, while keeping "
            "the rest of the texture the same. "
        )
        # Gemini API 호출
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[text_input, image],
            config=GenerateContentConfig(response_modalities=["Text", "Image"]),
        )

        # 응답 처리
        first_image_data = None

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                first_image_data = part.inline_data.data
                break  # 첫 번째 파일만 가져옴

        if not first_image_data:
            logger.debug("생성된 이미지가 없습니다.")
            return AssetResponse(success=False)

        # 파일 저장
        file_name = f"{asset_model}_{asset_type}_{datetime.now()}.jpg"
        file_path = os.path.join("app/assets/created", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(first_image_data)

        image_base64 = base64.b64encode(first_image_data).decode("utf-8")
        file_info = AssetFile(
            filename=file_name,
            content_base64=image_base64,
            size=len(first_image_data),
        )

        return AssetResponse(success=True, file=file_info)

    except Exception as e:
        logger.debug(f"에셋 생성 중 오류 발생: {e}")
        return AssetResponse(success=False)
