"""
리뷰 데이터 처리 서비스
=========================
Author: Jin
Date: 2025.09.30
Version: 1.0

Description:
리뷰 데이터의 로딩부터 분석까지 전체 파이프라인을 제공하는 서비스입니다.
ReviewDataLoader와 ReviewDataAnalyzer를 조합하여 사용합니다.
"""
import pandas as pd
from typing import Dict, Any

from base.service_base import ServiceBase
from services.review.review_data_loader import ReviewDataLoader
from services.review.review_data_analyzer import ReviewDataAnalyzer
from models.review_model import ReviewDataBatch
from config.logging_config import logger


class ReviewService(ServiceBase):
    """리뷰 데이터 처리 서비스"""
    
    def __init__(self):
        """
        리뷰 데이터 처리 서비스 초기화
        
        ReviewDataLoader와 ReviewDataAnalyzer를 조합하여
        리뷰 데이터의 전체 처리 파이프라인을 제공합니다.
        """
        loader = ReviewDataLoader()
        analyzer = ReviewDataAnalyzer()
        super().__init__(
            name="ReviewService",
            loader=loader,
            analyzer=analyzer
        )
        self.review_batch: ReviewDataBatch | None = None
    
    def validate_data(self, filepath: str) -> bool:
        """
        리뷰 CSV인지 검증 (필수 컬럼 확인)
        
        Args:
            filepath: 검증할 파일 경로
            
        Returns:
            리뷰 데이터 여부
        """
        try:
            # 헤더만 읽어서 컬럼 확인
            df_sample = pd.read_csv(filepath, nrows=0)
            columns = set(df_sample.columns.str.lower())
            
            # 리뷰 데이터 필수 컬럼 (조건 완화)
            review_indicators = {'review', 'rating', 'content', 'comment', 'text'}
            
            has_review_column = bool(columns & review_indicators)
            
            if has_review_column:
                logger.info(f"[{self.name}] 리뷰 데이터로 감지됨")
                return True
            
            logger.warning(f"[{self.name}] 리뷰 데이터가 아님: {columns}")
            return False
            
        except Exception as e:
            logger.error(f"[{self.name}] 검증 실패: {e}")
            return False
    
    def load(self, filepath: str) -> ReviewDataBatch:
        """
        리뷰 데이터 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 리뷰 데이터 배치
        """
        logger.info(f"[{self.name}] 리뷰 데이터 로딩 중...")
        self.review_batch = self.loader.process(filepath) 
        return self.review_batch
    
    def analyze(self, data: ReviewDataBatch) -> Dict[str, Any]:
        """
        리뷰 데이터 분석
        
        Args:
            data: 분석할 리뷰 데이터 배치
            
        Returns:
            분석 결과 딕셔너리
        """
        logger.info(f"[{self.name}] 리뷰 분석 중...")
        
        # Analyzer에 데이터 로드
        self.analyzer.load_review_data(data) 
        
        # 모든 분석 실행
        results = {
            'basic_statistics': self.analyzer.basic_statistics(),
            'correlation_analysis': self.analyzer.correlation_analysis(),
            'outlier_detection': self.analyzer.outlier_detection(),
            'domain_analysis': self.analyzer.domain_specific_analysis()
        }
        
        return results