"""
보험 데이터 모델
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
보험 데이터 전용 모델 클래스입니다.
동적 필드를 지원하는 DataModelBase를 상속받아 보험 특화 기능을 제공하며,
배치 처리, DataFrame 변환, 통계 정보 등의 기능을 포함합니다.
"""
from typing import List
import pandas as pd
from base.model_base import DataModelBase as InsuranceData


class InsuranceDataBatch:
    """보험 데이터 배치 처리 클래스"""
    
    def __init__(self, insurances: List[InsuranceData] = None):
        self.insurances = insurances if insurances is not None else []
    
    def add_insurance(self, insurance: InsuranceData):
        """
        보험 데이터 추가
        
        Args:
            insurance: 추가할 보험 데이터 객체
        """
        self.insurances.append(insurance)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        DataFrame으로 변환
        
        Returns:
            보험 데이터를 포함한 DataFrame
        """
        if not self.insurances:
            return pd.DataFrame()
        
        data = [insurance.to_dict() for insurance in self.insurances]
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'InsuranceDataBatch':
        """
        DataFrame에서 생성 - 컬럼명을 필드명으로 직접 사용
        
        Args:
            df: 변환할 DataFrame
            
        Returns:
            생성된 InsuranceDataBatch 객체
        """
        if df.empty:
            return cls([])
        
        print(f"DataFrame 컬럼: {list(df.columns)}")
        
        insurances = []
        for _, row in df.iterrows():
            # DataFrame의 모든 컬럼을 InsuranceData의 필드로 직접 사용
            insurance_data = {}
            
            for column in df.columns:
                value = row[column]
                
                # None/NaN 값 처리
                if pd.isna(value):
                    insurance_data[column] = None
                else:
                    insurance_data[column] = value
            
            insurance = InsuranceData(**insurance_data)
            insurances.append(insurance)
        
        print(f"생성된 InsuranceData 필드 예시: {insurances[0].get_fields()}")
        return cls(insurances)
    
    @property
    def size(self) -> int:
        """
        배치 크기
        
        Returns:
            보험 데이터 개수
        """
        return len(self.insurances)
    
    def __repr__(self) -> str:
        """
        객체의 문자열 표현 생성
        
        Returns:
            배치 크기를 포함한 문자열
        """
        return f"InsuranceDataBatch(size={self.size})"
    
    def __len__(self) -> int:
        """
        객체의 길이 반환
        
        Returns:
            배치 크기
        """
        return self.size