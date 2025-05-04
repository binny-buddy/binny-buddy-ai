from pydantic import BaseModel


class UnprocessableEntityResponse(BaseModel):
    """422 Unprocessable Entity 응답 모델"""

    detail: str
    error_code: int = 422
    error_message: str = "Unprocessable Entity"


class TooManyRequestsResponse(BaseModel):
    """429 Too Many Requests 응답 모델"""

    detail: str
    error_code: int = 429
    error_message: str = "Too Many Requests"
