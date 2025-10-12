"""
검색기 베이스 클래스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
사용자 쿼리에 대해 관련 문서를 검색하는 베이스 클래스입니다.
임베딩 모델과 벡터 저장소를 활용하여 의미 기반 검색을 수행합니다.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from config.logging_config import logger


class SearchResult:
    """검색 결과 데이터 클래스"""
    
    def __init__(
        self,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ):
        """
        검색 결과 객체 초기화
        
        Args:
            content: 검색된 문서 내용
            score: 유사도 점수
            metadata: 문서 메타데이터
            doc_id: 문서 ID
        """
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.doc_id = doc_id
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            문서 ID와 유사도 점수를 포함한 문자열
        """
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"SearchResult(id={self.doc_id}, score={self.score:.4f}, content='{preview}')"


class RetrieverBase(ABC):
    """검색기 베이스 클래스"""
    
    def __init__(
        self,
        top_k: int = 5,
        name: Optional[str] = None
    ):
        """
        검색기 초기화
        
        Args:
            top_k: 반환할 결과 개수
            name: 검색기 이름
        """
        self.top_k = top_k
        self.name = name or self.__class__.__name__
        self.search_count = 0
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        쿼리에 대한 관련 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수 (None이면 기본값 사용)
            filter_dict: 메타데이터 필터링 조건
            
        Returns:
            검색 결과 리스트
        """
        pass
    
    @abstractmethod
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[SearchResult]]:
        """
        여러 쿼리에 대한 일괄 검색
        
        Args:
            queries: 검색 쿼리 리스트
            top_k: 반환할 결과 개수 (None이면 기본값 사용)
            
        Returns:
            각 쿼리에 대한 검색 결과 리스트
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        검색기 통계 정보 반환
        
        Returns:
            검색기명, top_k, 검색 횟수를 포함한 딕셔너리
        """
        return {
            'retriever_name': self.name,
            'top_k': self.top_k,
            'search_count': self.search_count
        }
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            검색기명과 설정을 포함한 문자열
        """
        return f"{self.name}[top_k={self.top_k}, searches={self.search_count}]"

