"""
문서 로더 베이스 클래스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
다양한 형식의 문서를 로드하는 베이스 클래스입니다.
PDF, TXT, DOCX, HTML 등 여러 문서 형식을 통일된 인터페이스로 처리할 수 있도록 설계되었습니다.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from config.logging_config import logger


class Document:
    """문서 데이터 클래스"""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        문서 객체 초기화
        
        Args:
            content: 문서 내용
            metadata: 문서 메타데이터 (파일명, 페이지 번호 등)
        """
        self.content = content
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            문서 내용 미리보기를 포함한 문자열
        """
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


class DocumentLoaderBase(ABC):
    """문서 로더 베이스 클래스 - 다양한 문서 형식 지원"""
    
    def __init__(self, name: Optional[str] = None):
        """
        문서 로더 초기화
        
        Args:
            name: 로더 이름
        """
        self.name = name or self.__class__.__name__
        self.loaded_count = 0
    
    def _validate_file(self, filepath: str) -> Path:
        """
        파일 경로 검증
        
        Args:
            filepath: 검증할 파일 경로
            
        Returns:
            검증된 Path 객체
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: 올바른 파일이 아닌 경우
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
        if not file_path.is_file():
            raise ValueError(f"올바른 파일이 아닙니다: {filepath}")
        return file_path
    
    @abstractmethod
    def load(self, filepath: str) -> List[Document]:
        """
        파일에서 문서 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 문서 리스트
        """
        pass
    
    @abstractmethod
    def load_batch(self, filepaths: List[str]) -> List[Document]:
        """
        여러 파일에서 문서를 일괄 로드
        
        Args:
            filepaths: 로드할 파일 경로 리스트
            
        Returns:
            로드된 문서 리스트
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        로더 통계 정보 반환
        
        Returns:
            로더명과 처리 횟수를 포함한 딕셔너리
        """
        return {
            'loader_name': self.name,
            'loaded_count': self.loaded_count
        }
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            로더명과 로드 횟수를 포함한 문자열
        """
        return f"{self.name}[loaded={self.loaded_count}]"

