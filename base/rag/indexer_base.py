"""
인덱서 베이스 클래스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
문서를 임베딩하여 벡터 저장소에 인덱싱하는 베이스 클래스입니다.
문서 로더, 전처리기, 임베딩 모델, 벡터 저장소를 조합하여 전체 인덱싱 파이프라인을 제공합니다.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from config.logging_config import logger


class IndexerBase(ABC):
    """인덱서 베이스 클래스"""
    
    def __init__(
        self,
        name: Optional[str] = None
    ):
        """
        인덱서 초기화
        
        Args:
            name: 인덱서 이름
        """
        self.name = name or self.__class__.__name__
        self.indexed_count = 0
        self.total_chunks = 0
    
    @abstractmethod
    def index_document(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        단일 문서를 인덱싱
        
        Args:
            filepath: 인덱싱할 문서 파일 경로
            metadata: 문서 메타데이터
            
        Returns:
            인덱싱 결과 (처리된 청크 수, ID 리스트 등)
        """
        pass
    
    @abstractmethod
    def index_documents(
        self,
        filepaths: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        여러 문서를 일괄 인덱싱
        
        Args:
            filepaths: 인덱싱할 문서 파일 경로 리스트
            metadata_list: 각 문서의 메타데이터 리스트
            
        Returns:
            인덱싱 결과 (총 처리된 청크 수, 성공/실패 개수 등)
        """
        pass
    
    
    def get_stats(self) -> Dict[str, Any]:
        """
        인덱서 통계 정보 반환
        
        Returns:
            인덱서명, 인덱싱된 문서 수, 총 청크 수를 포함한 딕셔너리
        """
        return {
            'indexer_name': self.name,
            'indexed_count': self.indexed_count,
            'total_chunks': self.total_chunks
        }
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            인덱서명과 인덱싱 정보를 포함한 문자열
        """
        return f"{self.name}[docs={self.indexed_count}, chunks={self.total_chunks}]"

