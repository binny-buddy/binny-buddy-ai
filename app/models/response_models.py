from typing import List, Optional

from pydantic import BaseModel


class DetectedObject(BaseModel):
    """객체 탐지 결과의 개별 객체 정보"""

    label: str
    confidence: float
    description: Optional[str] = None
    box_2d: Optional[List[float]] = (
        None  # [ymin, xmin, ymax, xmax] 좌표 (0-1000 정규화)
    )


class DetectionResponse(BaseModel):
    """객체 탐지 API 응답 모델"""

    success: bool
    objects: List[DetectedObject]
    total_objects: int
