"""
동적 데이터 모델 베이스 클래스
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
동적 필드를 지원하는 범용 데이터 모델 클래스입니다.
CSV 컬럼명을 필드명으로 하는 유연한 데이터 구조를 제공하며,
다양한 형태의 데이터를 표준화된 객체로 변환할 수 있습니다.
"""
from typing import Dict, List, Any


class DataModelBase:
    """동적 필드를 지원하는 데이터 클래스"""
    
    def __init__(self, **kwargs):
        """
        CSV 컬럼명을 필드명으로 하는 동적 데이터 클래스
        
        Args:
            **kwargs: CSV 컬럼명=값 형태의 키워드 인자들
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self._fields = list(kwargs.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리로 변환
        
        Returns:
            필드와 값을 포함한 딕셔너리
        """
        return {field: getattr(self, field) for field in self._fields}
    
    def get_fields(self) -> List[str]:
        """
        필드 목록 반환
        
        Returns:
            필드명 목록
        """
        return self._fields.copy()
    
    def has_field(self, field_name: str) -> bool:
        """
        특정 필드 존재 여부 확인
        
        Args:
            field_name: 확인할 필드명
            
        Returns:
            필드 존재 여부
        """
        return field_name in self._fields
    
    def get_field_value(self, field_name: str, default: Any = None) -> Any:
        """
        필드 값 안전하게 가져오기
        
        Args:
            field_name: 가져올 필드명
            default: 필드가 없을 때 기본값
            
        Returns:
            필드 값 또는 기본값
        """
        return getattr(self, field_name, default)
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            필드와 값을 포함한 문자열
        """
        field_strs = [f"{field}={getattr(self, field)}" for field in self._fields[:]] 
        if len(self._fields) > 3:
            field_strs.append("...")
        return f"Data({', '.join(field_strs)})"
    
    def __str__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            필드 수와 필드 목록을 포함한 문자열
        """
        return f"Data with {len(self._fields)} fields: {self._fields}"
