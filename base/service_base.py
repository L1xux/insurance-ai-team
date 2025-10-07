"""
범용 서비스 베이스 클래스
=========================
Author: Jin
Date: 2025.09.30
Version: 1.0

Description:
Loader와 Analyzer를 조합하여 전체 데이터 처리 파이프라인을 제공하는 베이스 클래스입니다.
도메인별 서비스들이 이를 상속받아 확장할 수 있도록 설계되었습니다.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from base.data_loader_base import DataLoaderBase
from base.data_analyzer_base import DataAnalyzerBase
from config.logging_config import logger


class ServiceBase(ABC):
    """범용 서비스 베이스 클래스 - Loader와 Analyzer를 조합"""    
    def __init__(
        self, 
        name: Optional[str] = None,
        loader: Optional[DataLoaderBase] = None,
        analyzer: Optional[DataAnalyzerBase] = None
    ):
        self.name = name or self.__class__.__name__
        self.loader = loader
        self.analyzer = analyzer
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    def validate_data(self, filepath: str) -> bool:
        """
        데이터 형식 검증 (CSV가 해당 도메인에 맞는지 확인)
        
        Args:
            filepath: 검증할 파일 경로
            
        Returns:
            검증 성공 여부
        """
        pass
    
    def execute(self, filepath: str) -> Dict[str, Any]:
        """
        전체 파이프라인 실행: 검증 → 로드 → 분석
        
        Args:
            filepath: 처리할 파일 경로
            
        Returns:
            분석 결과
        """
        logger.info(f"[{self.name}] 파이프라인 시작: {filepath}")
        
        # 1. 데이터 검증
        if not self.validate_data(filepath):
            raise ValueError(f"[{self.name}] 데이터 검증 실패: {filepath}")
        
        # 2. 데이터 로드
        data = self.load(filepath)
        
        # 3. 데이터 분석
        analysis_results = self.analyze(data)
        
        # 4. 결과 저장
        self.results = {
            'filepath': str(filepath),
            'data_info': self.loader.get_data_info() if self.loader else {},
            'analysis': analysis_results,
            'status': 'success'
        }
        
        logger.info(f"[{self.name}] 파이프라인 완료")
        return self.results
    
    @abstractmethod
    def load(self, filepath: str) -> Any:
        """
        도메인별 데이터 로딩 로직
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            도메인별 데이터 객체
        """
        pass
    
    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        도메인별 분석 로직
        
        Args:
            data: 분석할 데이터
            
        Returns:
            분석 결과
        """
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """
        분석 결과 반환
        
        Returns:
            분석 결과 딕셔너리
        """
        return self.results
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            서비스 이름과 실행 상태를 포함한 문자열
        """
        status = "executed" if self.results else "ready"
        return f"{self.name}[{status}]"