from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class PlasticType(str, Enum):
    cup = "cup"
    bottle = "bottle"
    container = "container"


class WasteStatus(str, Enum):
    clean = "clean"
    dirty = "dirty"


class DetectedObject(BaseModel):
    """객체 탐지 결과의 개별 객체 정보"""

    label: PlasticType
    confidence: float
    status: WasteStatus
    how_to_recycle: Optional[str] = None
    box_2d: Optional[List[float]]  # [ymin, xmin, ymax, xmax] 좌표


class DetectionResponse(BaseModel):
    """객체 탐지 API 응답 모델"""

    success: bool
    objects: List[DetectedObject]
    total_objects: int
