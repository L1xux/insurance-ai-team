"""
리뷰 데이터 분석 서비스
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
리뷰 데이터에 특화된 분석 기능을 제공하는 서비스입니다.
카테고리별 분석, 평점 분석, 감정 분석 등 리뷰 도메인 특화 분석을 수행하며,
DataAnalyzerBase를 상속받아 확장된 기능을 제공합니다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from collections import Counter
import re

from base.data_analyzer_base import DataAnalyzerBase
from models.review_model import ReviewDataBatch
from config.logging_config import logger
from utils.profiler import measure_performance


class ReviewDataAnalyzer(DataAnalyzerBase):
    """리뷰 데이터 전용 분석기 - DataAnalyzerBase 상속"""
    
    def __init__(self):
        """
        리뷰 데이터 분석기 초기화
        
        DataAnalyzerBase를 상속받아 리뷰 데이터 전용 분석기를 생성합니다.
        """
        super().__init__("ReviewDataAnalyzer")
        self.review_batch: Optional[ReviewDataBatch] = None
    
    def load_review_data(self, review_batch: ReviewDataBatch) -> None:
        """
        리뷰 데이터 배치 로드
        
        Args:
            review_batch: 로드할 리뷰 데이터 배치
            
        Raises:
            ValueError: 유효하지 않은 ReviewDataBatch인 경우
        """
        if review_batch is None or review_batch.size == 0:
            raise ValueError("유효하지 않은 ReviewDataBatch입니다.")
        
        self.review_batch = review_batch
        df = review_batch.to_dataframe()
        self.load_data(df)
        
        logger.info(f"[{self.name}] 리뷰 데이터 로드: {review_batch.size}개 리뷰")
    
    def domain_specific_analysis(self) -> Dict[str, Any]:
        """리뷰 도메인 특화 분석"""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        results = {}
        
        # 1. 리뷰 길이 분석
        if 'review_length' in self.df.columns:
            results['review_length_analysis'] = self._analyze_review_length()
        
        # 2. 감정 점수 분석
        if 'sentiment_score' in self.df.columns:
            results['sentiment_analysis'] = self._analyze_sentiment()
        
        # 3. 평점 분석
        if 'rating' in self.df.columns:
            results['rating_analysis'] = self._analyze_ratings()
        
        # 4. 카테고리별 분석
        if 'category' in self.df.columns:
            results['category_analysis'] = self._analyze_by_category()
        
        
        self.analysis_results['review_domain_analysis'] = results
        logger.info(f"[{self.name}] 리뷰 도메인 분석 완료")
        return results
    
    def _analyze_review_length(self) -> Dict[str, Any]:
        """
        리뷰 길이 분석
        
        Returns:
            리뷰 길이 분석 결과
        """
        length_col = self.df['review_length'] # type: ignore
        
        return {
            'statistics': {
                'mean': length_col.mean(),
                'median': length_col.median(),
                'std': length_col.std(),
                'min': length_col.min(),
                'max': length_col.max()
            },
            'length_categories': {
                'short_reviews': (length_col < 50).sum(),  # 50자 미만
                'medium_reviews': ((length_col >= 50) & (length_col < 200)).sum(),  # 50-200자
                'long_reviews': (length_col >= 200).sum()  # 200자 이상
            }
        }
    
    def _analyze_sentiment(self) -> Dict[str, Any]:
        """
        감정 분석
        
        Returns:
            감정 분석 결과
        """
        sentiment_col = self.df['sentiment_score'] # type: ignore
        
        # 감정 카테고리 분류
        positive = (sentiment_col > 0.1).sum()
        neutral = ((sentiment_col >= -0.1) & (sentiment_col <= 0.1)).sum()
        negative = (sentiment_col < -0.1).sum()
        
        return {
            'statistics': {
                'mean': sentiment_col.mean(),
                'median': sentiment_col.median(),
                'std': sentiment_col.std(),
                'min': sentiment_col.min(),
                'max': sentiment_col.max()
            },
            'sentiment_distribution': {
                'positive': positive,
                'neutral': neutral,
                'negative': negative,
                'positive_ratio': positive / len(sentiment_col),
                'negative_ratio': negative / len(sentiment_col)
            }
        }
    
    def _analyze_ratings(self) -> Dict[str, Any]:
        """
        평점 분석
        
        Returns:
            평점 분석 결과
        """
        rating_col = self.df['rating'] # type: ignore
        
        rating_distribution = rating_col.value_counts().sort_index()
        
        return {
            'statistics': {
                'mean': rating_col.mean(),
                'median': rating_col.median(),
                'mode': rating_col.mode().iloc[0] if not rating_col.mode().empty else None,
                'std': rating_col.std()
            },
            'distribution': rating_distribution.to_dict(),
            'high_ratings_ratio': (rating_col >= 4).sum() / len(rating_col),
            'low_ratings_ratio': (rating_col <= 2).sum() / len(rating_col)
        }
    
    def _analyze_by_category(self) -> Dict[str, Any]:
        """
        카테고리별 분석
        
        Returns:
            카테고리별 분석 결과
        """
        category_stats = {}
        
        for category in self.df['category'].unique(): # type: ignore
            category_data = self.df[self.df['category'] == category] # type: ignore
            
            stats = {
                'count': len(category_data),
                'percentage': len(category_data) / len(self.df) * 100 # type: ignore
            }
            
            # 평점 정보
            if 'rating' in category_data.columns:
                stats['avg_rating'] = category_data['rating'].mean()
            
            # 감정 정보
            if 'sentiment_score' in category_data.columns:
                stats['avg_sentiment'] = category_data['sentiment_score'].mean()
            
            # 리뷰 길이 정보
            if 'review_length' in category_data.columns:
                stats['avg_length'] = category_data['review_length'].mean()
            
            category_stats[category] = stats
        
        return {
            'category_statistics': category_stats,
            'total_categories': len(category_stats)
        }
    
