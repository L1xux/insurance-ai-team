"""
리뷰 데이터 모델
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
리뷰 데이터 전용 모델 클래스입니다.
동적 필드를 지원하는 BaseModel을 상속받아 리뷰 특화 기능을 제공하며,
배치 처리, DataFrame 변환, 통계 정보 등의 기능을 포함합니다.
"""
from typing import List
import pandas as pd
from base.model_base import DataModelBase as ReviewData

class ReviewDataBatch:
    """리뷰 데이터 배치 처리 클래스"""
    
    def __init__(self, reviews: List[ReviewData] = []):
        self.reviews = reviews
    
    def add_review(self, review: ReviewData):
        """
        리뷰 추가
        
        Args:
            review: 추가할 리뷰 데이터 객체
        """
        self.reviews.append(review)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        DataFrame으로 변환
        
        Returns:
            리뷰 데이터를 포함한 DataFrame
        """
        if not self.reviews:
            return pd.DataFrame()
        
        data = [review.to_dict() for review in self.reviews]
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'ReviewDataBatch':
        """
        DataFrame에서 생성 - 컬럼명을 필드명으로 직접 사용
        
        Args:
            df: 변환할 DataFrame
            
        Returns:
            생성된 ReviewDataBatch 객체
        """
        if df.empty:
            return cls([])
        
        print(f"DataFrame 컬럼: {list(df.columns)}")
        
        reviews = []
        for _, row in df.iterrows():
            # DataFrame의 모든 컬럼을 ReviewData의 필드로 직접 사용
            review_data = {}
            
            for column in df.columns:
                value = row[column]
                
                # None/NaN 값 처리
                if pd.isna(value):
                    review_data[column] = None
                else:
                    review_data[column] = value
            
            review = ReviewData(**review_data)
            reviews.append(review)
        
        print(f"생성된 ReviewData 필드 예시: {reviews[0].get_fields()}")
        return cls(reviews)
    
    @property
    def size(self) -> int:
        """
        배치 크기
        
        Returns:
            리뷰 데이터 개수
        """
        return len(self.reviews)
    