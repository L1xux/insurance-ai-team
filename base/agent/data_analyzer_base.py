"""
범용 데이터 분석기 베이스 클래스
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
모든 데이터 분석기의 기본이 되는 추상 베이스 클래스입니다.
기본 통계 분석, 데이터 로딩, 시각화 스타일 설정 등 공통 기능을 제공하며,
도메인별 분석기들이 이를 상속받아 확장할 수 있도록 설계되었습니다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from config.logging_config import logger
from config.settings import settings

class DataAnalyzerBase(ABC):
    """범용 데이터 분석기 베이스 클래스"""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.df: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}
        
        # 플롯 스타일 설정
        try:
            plt.style.use(settings.plot_style)
        except (OSError, ValueError) as e:
            logger.warning(f"스타일 '{settings.plot_style}' 로드 실패, 기본 스타일 사용: {e}")
            plt.style.use('default')
        
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = settings.figure_size
    
    def load_data(self, df: pd.DataFrame) -> None:
        """
        분석할 데이터 로드
        
        Args:
            df: 분석할 DataFrame
        """
        if df is None or df.empty:
            raise ValueError("유효하지 않은 DataFrame입니다.")
        
        self.df = df.copy()
        logger.info(f"[{self.name}] 분석 데이터 로드: {self.df.shape}")
    
    def basic_statistics(self) -> Dict[str, Any]:
        """
        기본 통계 분석
        
        Returns:
            기본 통계 분석 결과 딕셔너리
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        results = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # 숫자형 컬럼 통계
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results['numeric_summary'] = self.df[numeric_cols].describe().to_dict()
        
        # 카테고리형 컬럼 통계
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            results['categorical_summary'][col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head().to_dict(),
                'missing_count': self.df[col].isnull().sum()
            }
        
        self.analysis_results['basic_statistics'] = results
        logger.info(f"[{self.name}] 기본 통계 분석 완료")
        return results
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """
        상관관계 분석
        
        Returns:
            상관관계 분석 결과 딕셔너리
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logger.warning(f"[{self.name}] 숫자형 컬럼이 없어 상관관계 분석을 건너뜁니다.")
            return {}
        
        correlation_matrix = numeric_df.corr()
        # 강한 상관관계 찾기
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                
                if abs(corr_val) >= 0.7: # type: ignore
                    strong_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        results = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
        }
        
        self.analysis_results['correlation_analysis'] = results
        logger.info(f"결과: {results} [{self.name}] 상관관계 분석 완료")
        return results
    
    def outlier_detection(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        이상치 탐지
        
        Args:
            columns: 분석할 컬럼 목록 (None이면 숫자형 컬럼 전체)
            
        Returns:
            이상치 탐지 결과 딕셔너리
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            series = self.df[col].dropna()
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (series < lower_bound) | (series > upper_bound)
            outliers_series: pd.Series = series[outliers_mask]
            outlier_values = outliers_series.tolist()[:10] if len(outliers_series) > 0 else []

            outliers_info[col] = {
                'outlier_count': len(outliers_series),
                'outlier_percentage': len(outliers_series) / len(series) * 100,
                'outlier_values': outlier_values  
            }
        
        results = {
            'outliers_by_column': outliers_info,
            'total_outliers': sum(info['outlier_count'] for info in outliers_info.values() if isinstance(info['outlier_count'], int))
        }
        
        self.analysis_results['outlier_detection'] = results
        logger.info(f"[{self.name}] 이상치 탐지 완료")
        return results
    
    @abstractmethod
    def domain_specific_analysis(self) -> Dict[str, Any]:
        """
        도메인별 특화된 분석 로직 (하위 클래스에서 구현)
        
        Returns:
            도메인별 분석 결과
        """
        pass
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            분석기 이름, 데이터 상태, 분석 결과 수를 포함한 문자열
        """
        data_status = f"loaded ({self.df.shape})" if self.df is not None else "empty"
        analyses_count = len(self.analysis_results)
        return f"{self.name}[data: {data_status}, analyses: {analyses_count}]"
