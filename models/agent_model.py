"""
Agent 데이터 모델
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
Agent 관련 모든 데이터 모델입니다.
- dataclass: 간단한 데이터 구조
- Pydantic:  structured output용
"""
from dataclasses import dataclass
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field


# ========== Dataclass Models ==========

@dataclass
class AgentResult:
    """Agent 실행 결과"""
    success: bool
    data: Any
    error: Optional[str] = None


# ========== Pydantic Models ==========

class ToolPlan(BaseModel):
    """Tool 실행 계획"""
    tool_name: str = Field(description="사용할 Tool 이름")
    query: Optional[str] = Field(default=None, description="검색 쿼리 (RAG 도구용)")
    analysis_type: Optional[str] = Field(default=None, description="분석 유형 (분석 도구용)")
    top_k: Optional[int] = Field(default=None, description="반환 개수")


class AgentPlan(BaseModel):
    """Worker Agent 실행 계획"""
    reasoning: str = Field(description="계획 수립 이유 및 분석")
    tools: List[ToolPlan] = Field(default_factory=list, description="실행할 Tool 목록 (순서대로)")


class TaskPlan(BaseModel):
    """Manager의 작업 계획"""
    task_id: str = Field(description="작업 고유 ID")
    agent_name: str = Field(description="할당할 Worker Agent 이름")
    description: str = Field(description="작업 설명")
    query: Optional[str] = Field(default=None, description="작업에 필요한 쿼리")


class ExecutionPlan(BaseModel):
    """Manager의 전체 실행 계획"""
    analysis: str = Field(description="사용자 요청 분석")
    tasks: List[TaskPlan] = Field(default_factory=list, description="작업 계획 리스트")
