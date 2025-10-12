"""
Tool 데이터 모델
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
Tool 관련 데이터 모델입니다.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class ToolResult:
    """Tool 실행 결과"""
    success: bool
    data: Any
    error: Optional[str] = None


@dataclass
class ToolSchema:
    """Tool 스키마"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'type': 'object',
                'properties': self.parameters,
                'required': self.required_params
            }
        }

