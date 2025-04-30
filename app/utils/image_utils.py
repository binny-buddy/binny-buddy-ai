import io
from typing import Optional, Tuple

from PIL import Image


def get_image_format(image_data: bytes) -> Optional[str]:
    """
    이미지 데이터의 포맷을 확인하는 함수

    Args:
        image_data: 이미지 바이너리 데이터

    Returns:
        Optional[str]: 이미지 포맷 (jpeg, png 등) 또는 None
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        return img.format.lower() if img.format else None
    except Exception:
        return None


def validate_image(image_data: bytes) -> bool:
    """
    이미지 데이터가 유효한지 검증하는 함수

    Args:
        image_data: 이미지 바이너리 데이터

    Returns:
        bool: 이미지가 유효하면 True, 아니면 False
    """
    try:
        img_format = get_image_format(image_data)

        # 이미지 포맷 확인
        if img_format not in ["jpeg", "png", "jpg", "bmp"]:
            return False

        img = Image.open(io.BytesIO(image_data))
        img.verify()  # 이미지 데이터 검증
        return True
    except Exception:
        return False


def resize_image(image_data: bytes, max_size: Tuple[int, int] = (1024, 1024)) -> bytes:
    """
    이미지를 지정된 최대 크기로 리사이징하는 함수

    Args:
        image_data: 이미지 바이너리 데이터
        max_size: 최대 크기 (width, height)

    Returns:
        bytes: 리사이징된 이미지 바이너리 데이터
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail(max_size, Image.LANCZOS)

        output = io.BytesIO()
        img.save(output, format=img.format)
        return output.getvalue()
    except Exception:
        # 리사이징 실패 시 원본 이미지 반환
        return image_data
