from enum import Enum
from typing import Optional

from pydantic import BaseModel


class AssetType(str, Enum):
    texture = "texture"
    accessory = "accessory"


class AssetFile(BaseModel):
    filename: str
    content_base64: str
    size: Optional[int] = None


class AssetResponse(BaseModel):
    success: bool
    file: AssetFile | None = None
