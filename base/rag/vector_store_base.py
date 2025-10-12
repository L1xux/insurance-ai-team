"""
벡터 저장소 베이스 클래스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
임베딩 벡터를 저장하고 검색하는 벡터 데이터베이스의 베이스 클래스입니다.
FAISS, Chroma, Pinecone 등 다양한 벡터 저장소를 통일된 인터페이스로 사용할 수 있도록 설계되었습니다.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from config.logging_config import logger


class VectorStoreBase(ABC):
    """벡터 저장소 베이스 클래스 - 임베딩 벡터 저장 및 검색"""
    
    def __init__(
        self,
        dimension: int,
        name: Optional[str] = None
    ):
        """
        벡터 저장소 초기화
        
        Args:
            dimension: 벡터 차원 수
            name: 저장소 이름
        """
        self.dimension = dimension
        self.name = name or self.__class__.__name__
        self.vector_count = 0
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        벡터를 저장소에 추가
        
        Args:
            vectors: 추가할 벡터 리스트
            metadata: 각 벡터의 메타데이터 리스트
            ids: 각 벡터의 고유 ID 리스트
            
        Returns:
            추가된 벡터의 ID 리스트
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        쿼리 벡터와 유사한 벡터 검색
        
        Args:
            query_vector: 검색할 쿼리 벡터
            top_k: 반환할 결과 개수
            filter_dict: 메타데이터 필터링 조건
            
        Returns:
            (ID, 유사도 점수, 메타데이터) 튜플 리스트
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        벡터 삭제
        
        Args:
            ids: 삭제할 벡터 ID 리스트
            
        Returns:
            삭제 성공 여부
        """
        pass
    
    @abstractmethod
    def update(
        self,
        ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        벡터 또는 메타데이터 업데이트
        
        Args:
            ids: 업데이트할 벡터 ID 리스트
            vectors: 새로운 벡터 리스트
            metadata: 새로운 메타데이터 리스트
            
        Returns:
            업데이트 성공 여부
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> bool:
        """
        저장소를 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로
            
        Returns:
            저장 성공 여부
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> bool:
        """
        파일에서 저장소 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드 성공 여부
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        저장소 통계 정보 반환
        
        Returns:
            저장소명, 차원, 벡터 개수를 포함한 딕셔너리
        """
        return {
            'store_name': self.name,
            'dimension': self.dimension,
            'vector_count': self.vector_count
        }
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            저장소명과 벡터 정보를 포함한 문자열
        """
        return f"{self.name}[dim={self.dimension}, count={self.vector_count}]"

