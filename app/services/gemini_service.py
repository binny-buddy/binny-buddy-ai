import json
import os
from io import BytesIO
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentResponse,
)
from PIL import Image

from app.models.response_models import DetectedObject, DetectionResponse

# 환경 변수 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# Gemini 클라이언트 설정
client = genai.Client(api_key=GEMINI_API_KEY)


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
        "Detect all objects in the image. "
        'Describe the plastic is "clean" or "dirty" '
        "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000. "
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
    """
    Gemini API 응답을 파싱하여 탐지된 객체 목록을 생성합니다.

    Args:
        response: Gemini API 응답

    Returns:
        List[DetectedObject]: 탐지된 객체 목록
    """
    detected_objects = []

    if not response.candidates or not response.candidates[0].content.parts:
        return detected_objects

    # 텍스트 응답 추출
    try:
        response_text = response.candidates[0].content.parts[0].text

        # JSON 형식 응답 추출 시도
        import re

        json_match = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
        if json_match:
            objects_list = json.loads(json_match.group())

            for obj in objects_list:
                detected_objects.append(
                    DetectedObject(
                        label=obj.get("label", "unknown"),
                        confidence=obj.get("confidence", 1.0),
                        description=obj.get("description", ""),
                        box_2d=obj.get("box_2d"),
                    )
                )
        else:
            # 단순 텍스트 응답인 경우
            detected_objects.append(
                DetectedObject(
                    label="detected_object",
                    confidence=1.0,
                    description=(
                        response_text[:200]
                        if len(response_text) > 200
                        else response_text
                    ),
                )
            )
    except Exception as e:
        # 파싱 실패 시 에러 메시지 포함 객체 생성
        detected_objects.append(
            DetectedObject(
                label="parsing_error",
                confidence=0.0,
                description=f"응답 파싱 오류: {str(e)}",
            )
        )

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
        return create_detection_response(True, detected_objects)

    except Exception as e:
        # 오류 응답 생성
        return create_detection_response(False, [], str(e))
