"""
범용 데이터 로더 베이스 클래스
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
모든 데이터 로더의 기본이 되는 추상 베이스 클래스입니다.
CSV, JSON, Excel 등 다양한 형식의 데이터를 로드하는 공통 기능을 제공하며,
도메인별 로더들이 이를 상속받아 확장할 수 있도록 설계되었습니다.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from config.logging_config import logger

class DataLoaderBase(ABC):
    """범용 데이터 로더 베이스 클래스"""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.df: Optional[pd.DataFrame] = None
        self.processed_count = 0
        self.file_info: Dict[str, Any] = {}
    
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
    
    def load_csv(self, filepath: str, **kwargs: Any) -> pd.DataFrame:
        """
        CSV 파일을 로드
        
        Args:
            filepath: 로드할 CSV 파일 경로
            **kwargs: pandas.read_csv에 전달할 추가 인자
            
        Returns:
            로드된 DataFrame
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            Exception: 로딩 실패 시
        """
        try:
            file_path = self._validate_file(filepath)
            
            logger.info(f"[{self.name}] 데이터 로딩 중: {filepath}")
            
            # 기본 pandas 읽기 + 자동 타입 추론
            self.df = pd.read_csv(filepath, **kwargs)
            self.df = self.df.convert_dtypes() # type: ignore
            
            # 파일 정보 저장
            self.file_info = {
                'filepath': str(file_path),
                'filename': file_path.name,
                'size_bytes': file_path.stat().st_size,
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict()
            }
            
            logger.info(f"[{self.name}] 데이터 로드 완료: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            logger.info(f"[{self.name}] 자동 감지된 컬럼 타입:\n{self.df.dtypes}")
            logger.info(f"[{self.name}] 첫 3개 샘플:\n{self.df.head(3)}")
            
            self.processed_count += 1
            return self.df
            
        except Exception as e:
            logger.error(f"[{self.name}] 데이터 로딩 실패: {str(e)}")
            raise
    
    def load_excel(self, filepath: str, **kwargs: Any) -> pd.DataFrame:
        """
        Excel 파일 로드
        
        Args:
            filepath: 로드할 Excel 파일 경로
            **kwargs: pandas.read_excel에 전달할 추가 인자
            
        Returns:
            로드된 DataFrame
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            Exception: 로딩 실패 시
        """
        try:
            self._validate_file(filepath)
            
            logger.info(f"[{self.name}] Excel 데이터 로딩 중: {filepath}")
            
            self.df = pd.read_excel(filepath, **kwargs)
            self.df = self.df.convert_dtypes() # type: ignore
            
            logger.info(f"[{self.name}] Excel 데이터 로드 완료: {self.df.shape}")
            self.processed_count += 1
            return self.df
            
        except Exception as e:
            logger.error(f"[{self.name}] Excel 로딩 실패: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        데이터 정보 반환
        
        Returns:
            데이터 기본 정보, 품질 정보, 컬럼별 통계를 포함한 딕셔너리
        """
        if self.df is None:
            logger.warning(f"[{self.name}] 데이터가 로드되지 않았습니다.")
            return {}
        
        info: Dict[str, Any] = {
            'basic_info': self.file_info,
            'data_quality': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().to_dict(),
                'duplicated_rows': self.df.duplicated().sum(),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'column_stats': {}
        }
        
        # 컬럼별 기본 통계
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'non_null_count': self.df[col].count(),
                'null_count': self.df[col].isnull().sum(),
                'unique_count': self.df[col].nunique()
            }
            
            # 숫자형 컬럼의 경우 추가 통계
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info.update({
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                })
            
            info['column_stats'][col] = col_info
        
        return info
    
    def preview_data(self, rows: int = 5) -> pd.DataFrame:
        """
        데이터 미리보기
        
        Args:
            rows: 미리보기할 행 수 (기본값: 5)
            
        Returns:
            미리보기 DataFrame
        """
        if self.df is None:
            logger.warning(f"[{self.name}] 데이터가 로드되지 않았습니다.")
            return pd.DataFrame()
        
        return self.df.head(rows)
    
    @abstractmethod
    def process(self, filepath: str) -> Any:
        """
        도메인별 특화된 처리 로직 (하위 클래스에서 구현)
        
        Args:
            filepath: 처리할 파일 경로
            
        Returns:
            도메인별 데이터 객체
        """
        pass
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            로더 이름과 데이터 상태를 포함한 문자열
        """
        status = f"loaded ({self.df.shape})" if self.df is not None else "empty"
        return f"{self.name}[{status}]"
