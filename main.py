# main.py · 나만의 요리 레시피 도우미 백엔드

import base64
import binascii
import json
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.5-flash-lite"

# 서비스 이름과 소개
APP_TITLE = "나만의 요리 레시피 도우미"
APP_ICON = "🍳"
APP_DESCRIPTION = (
    "만들고 싶은 요리 이름과 사진을 입력하면 "
    "재료와 조리 방법을 알려드립니다."
)

# AI에게 시킬 일
PROMPT = """
당신은 요리 레시피 안내 도우미입니다.

사용자가 입력한 요리 이름을 기준으로
요리에 필요한 재료와 조리 순서를 한국어로 알려 주세요.

사진은 음식이나 재료를 확인하는 데 참고하세요.
사진에서 확인되지 않은 것은 추측하지 마세요.

음식 이름이 불명확하면 임의로 다른 음식을 고르지 말고
음식 이름을 구체적으로 입력해 달라고 안내하세요.

사진만으로 식품의 신선도나 안전성을 확정하지 마세요.
재료의 익힘 등 필요한 조리상 주의사항을 알려 주세요.
"""

# 결과로 보여줄 항목
FIELDS = {
    "음식": "사용자가 만들고 싶은 요리 이름",
    "사진에서 확인한 내용": (
        "사진에서 확실하게 확인할 수 있는 음식 또는 재료. "
        "불확실하면 확인하기 어렵다고 표시"
    ),
    "재료": (
        "요리에 필요한 재료. 사진으로 확인한 재료와 "
        "추가로 필요한 재료를 구분"
    ),
    "레시피": "따라 하기 쉽게 번호를 붙여 설명한 조리 순서",
    "주의사항": "조리와 재료 취급 시 주의할 점",
}

SCHEMA = {
    "type": "object",
    "properties": {
        name: {
            "type": "string",
            "description": description,
        }
        for name, description in FIELDS.items()
    },
    "required": list(FIELDS),
}

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
)


class PhotoRequest(BaseModel):
    image_b64: str = Field(
        min_length=1,
        max_length=4_000_000,
        description="JPEG 사진을 base64로 인코딩한 문자열",
    )
    food_name: str = Field(
        min_length=1,
        max_length=100,
        description="만들고 싶은 음식 이름",
    )


class Result(BaseModel):
    result: dict[str, str]
    tokens: int | None = None


def check_access(access_code):
    """접근 코드가 설정되어 있고 요청 값과 같은지 확인합니다."""
    expected = os.environ.get("APP_ACCESS_CODE")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="서버에 APP_ACCESS_CODE가 설정되지 않았습니다",
        )

    if access_code != expected:
        raise HTTPException(
            status_code=401,
            detail="접근 코드가 틀렸습니다",
        )


def get_client():
    """환경 변수에서 Gemini API 키를 읽습니다."""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="서버에 GEMINI_API_KEY가 설정되지 않았습니다",
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(
                http_status_codes=[500, 503]
            ),
        ),
    )


@app.get("/")
def home():
    """서비스 소개와 결과 항목을 반환합니다."""
    return {
        "name": APP_TITLE,
        "icon": APP_ICON,
        "description": APP_DESCRIPTION,
        "fields": list(FIELDS),
    }


@app.post("/analyze", response_model=Result)
def analyze(
    request: PhotoRequest,
    x_access_code: str | None = Header(default=None),
):
    # 접근 코드 확인
    check_access(x_access_code)

    # 음식 이름 확인
    food_name = request.food_name.strip()

    if not food_name:
        raise HTTPException(
            status_code=422,
            detail="음식 이름을 입력해 주세요",
        )

    # 사진의 base64 형식 확인
    try:
        photo = base64.b64decode(
            request.image_b64,
            validate=True,
        )
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422,
            detail="사진의 base64 형식이 올바르지 않습니다",
        )

    # JPEG 파일인지 확인
    if not photo.startswith(b"\xff\xd8\xff"):
        raise HTTPException(
            status_code=422,
            detail="JPEG 형식의 사진을 보내 주세요",
        )

    client = get_client()

    # 음식 이름과 사진을 Gemini에 전달
    try:
        reply = client.interactions.create(
            model=MODEL_NAME,
            store=False,
            input=[
                {
                    "type": "text",
                    "text": (
                        f"{PROMPT}\n\n"
                        f"사용자가 입력한 음식 이름: {food_name}"
                    ),
                },
                {
                    "type": "image",
                    "data": request.image_b64,
                    "mime_type": "image/jpeg",
                },
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": SCHEMA,
            },
        )
    except Exception as error:
        code = getattr(error, "status_code", None)

        raise HTTPException(
            status_code=429 if code == 429 else 502,
            detail="Gemini 요청에 실패했습니다. 잠시 뒤 다시 시도하세요.",
        ) from error

    # Gemini 응답을 JSON으로 변환
    try:
        result = json.loads(reply.output_text)

        if not isinstance(result, dict):
            raise ValueError("결과가 딕셔너리가 아닙니다")

        if any(
            not isinstance(result.get(name), str)
            for name in FIELDS
        ):
            raise ValueError("결과 항목이 올바르지 않습니다")

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="AI 응답을 읽을 수 없습니다. 다시 시도해 주세요",
        )

    usage = getattr(reply, "usage", None)
    tokens = getattr(usage, "total_tokens", None)

    return Result(
        result={name: result[name] for name in FIELDS},
        tokens=tokens,
    )
