"""Routes for the local AlphaAgent chatbot demonstration."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..models.api_models import ApiResponse
from ..services.demo import chat, get_job, get_overview


router = APIRouter(prefix="/api/demo", tags=["Demo Chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    date: Optional[str] = None


@router.get("/overview", response_model=ApiResponse[dict])
def overview():
    return ApiResponse(success=True, message="策略概览加载成功", data=get_overview())


@router.post("/chat", response_model=ApiResponse[dict])
def demo_chat(request: ChatRequest):
    try:
        return ApiResponse(success=True, message="消息处理成功", data=chat(request.message, request.date))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=ApiResponse[dict])
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启。")
    return ApiResponse(success=True, message="任务状态加载成功", data=job)
