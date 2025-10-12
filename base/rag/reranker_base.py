"""
재순위화 베이스 클래스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
검색된 문서의 순위를 재조정하는 베이스 클래스입니다.
초기 검색 결과를 더 정교한 모델이나 알고리즘을 사용하여 재순위화함으로써
검색 품질을 향상시킵니다.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from config.logging_config import logger


class RerankResult:
    """재순위화 결과 데이터 클래스"""
    
    def __init__(
        self,
        content: str,
        original_score: float,
        rerank_score: float,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        재순위화 결과 객체 초기화
        
        Args:
            content: 문서 내용
            original_score: 원본 유사도 점수
            rerank_score: 재순위화 점수
            metadata: 문서 메타데이터
            doc_id: 문서 ID
        """
        self.content = content
        self.original_score = original_score
        self.rerank_score = rerank_score
        self.metadata = metadata or {}
        self.doc_id = doc_id
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            문서 ID와 점수 정보를 포함한 문자열
        """
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"RerankResult(id={self.doc_id}, orig={self.original_score:.4f}, rerank={self.rerank_score:.4f})"


class RerankerBase(ABC):
    """재순위화 베이스 클래스 - 검색 결과 재정렬"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        name: Optional[str] = None
    ):
        """
        재순위화 모델 초기화
        
        Args:
            model_name: 사용할 재순위화 모델 이름
            top_k: 재순위화 후 반환할 결과 개수
            name: 재순위화기 이름
        """
        self.model_name = model_name
        self.top_k = top_k
        self.name = name or self.__class__.__name__
        self.rerank_count = 0
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        쿼리와 문서 리스트에 대해 재순위화 수행
        
        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트 (content, score, metadata 포함)
            top_k: 반환할 결과 개수 (None이면 기본값 사용)
            
        Returns:
            재순위화된 결과 리스트
        """
        pass
    
    @abstractmethod
    def compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        쿼리와 문서 간의 재순위화 점수 계산
        
        Args:
            query: 검색 쿼리
            documents: 점수를 계산할 문서 내용 리스트
            
        Returns:
            각 문서의 재순위화 점수 리스트
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        재순위화기 통계 정보 반환
        
        Returns:
            모델명, top_k, 재순위화 횟수를 포함한 딕셔너리
        """
        return {
            'reranker_name': self.name,
            'model_name': self.model_name,
            'top_k': self.top_k,
            'rerank_count': self.rerank_count
        }
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            재순위화기명과 모델 정보를 포함한 문자열
        """
        return f"{self.name}[model={self.model_name}, top_k={self.top_k}]"

