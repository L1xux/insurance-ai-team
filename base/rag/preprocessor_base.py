"""
전처리기 베이스 클래스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
문서를 청크(chunk)로 분할하고 전처리하는 베이스 클래스입니다.
텍스트 정제, 청킹, 메타데이터 추가 등의 전처리 작업을 수행합니다.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from config.logging_config import logger


class Chunk:
    """청크 데이터 클래스"""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None
    ):
        """
        청크 객체 초기화
        
        Args:
            content: 청크 내용
            metadata: 청크 메타데이터
            chunk_id: 청크 고유 ID
        """
        self.content = content
        self.metadata = metadata or {}
        self.chunk_id = chunk_id
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            청크 ID와 내용 미리보기를 포함한 문자열
        """
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(id={self.chunk_id}, content='{preview}')"


class PreprocessorBase(ABC):
    """전처리기 베이스 클래스"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        name: Optional[str] = None
    ):
        """
        전처리기 초기화
        
        Args:
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 겹침 크기
            name: 전처리기 이름
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.name = name or self.__class__.__name__
        self.processed_count = 0
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 텍스트 청크 리스트
        """
        pass
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        텍스트 전처리 (정제, 정규화 등)
        
        Args:
            text: 전처리할 텍스트
            
        Returns:
            전처리된 텍스트
        """
        pass
    
    def create_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        텍스트를 전처리하고 청크 객체로 변환
        
        Args:
            text: 처리할 텍스트
            metadata: 청크에 추가할 메타데이터
            
        Returns:
            청크 객체 리스트
        """
        # 전처리
        cleaned_text = self.preprocess(text)
        
        # 청크로 분할
        text_chunks = self.split_text(cleaned_text)
        
        # 청크 객체 생성
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = idx
            chunk_metadata['chunk_size'] = len(chunk_text)
            
            chunk = Chunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"{self.name}_{self.processed_count}_{idx}"
            )
            chunks.append(chunk)
        
        self.processed_count += 1
        logger.info(f"[{self.name}] 청크 생성 완료: {len(chunks)}개")
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """
        전처리기 통계 정보 반환
        
        Returns:
            전처리기 설정 및 처리 횟수를 포함한 딕셔너리
        """
        return {
            'preprocessor_name': self.name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'processed_count': self.processed_count
        }
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            전처리기명과 청크 설정을 포함한 문자열
        """
        return f"{self.name}[size={self.chunk_size}, overlap={self.chunk_overlap}]"

