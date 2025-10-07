"""
리뷰 데이터 로더 서비스
=========================
Author: Jin
Date: 2025.09.17
Version: 1.0

Description:
CSV 파일에서 리뷰 데이터를 로드하고 ReviewDataBatch 객체로 변환하는 서비스입니다.
DataLoaderBase를 상속받아 리뷰 도메인에 특화된 데이터 로딩 기능을 제공하며,
성능 측정 프로파일링이 적용되어 있습니다.
"""


from base.data_loader_base import DataLoaderBase
from models.review_model import ReviewDataBatch
from config.logging_config import logger


class ReviewDataLoader(DataLoaderBase):
    """리뷰 데이터 전용 로더 - DataLoaderBase 상속"""
    
    def __init__(self):
        """
        리뷰 데이터 로더 초기화
        
        DataLoaderBase를 상속받아 리뷰 데이터 전용 로더를 생성합니다.
        """
        super().__init__("ReviewDataLoader")
    
    def process(self, filepath: str) -> ReviewDataBatch:
        """
        리뷰 CSV 파일을 로드하여 ReviewDataBatch로 변환
        
        Args:
            filepath: 리뷰 CSV 파일 경로
            
        Returns:
            ReviewDataBatch: 리뷰 데이터 배치 객체
        """
        try:
            # 부모 클래스의 CSV 로딩 사용
            df = self.load_csv(filepath)
            
            # ReviewDataBatch로 변환
            review_batch = ReviewDataBatch.from_dataframe(df)
            
            logger.info(f"[{self.name}] ReviewDataBatch 생성 완료: {review_batch.size}개 리뷰")
            return review_batch
            
        except Exception as e:
            logger.error(f"[{self.name}] 리뷰 데이터 처리 실패: {str(e)}")
            raise
    

