"""
보험 데이터 처리 서비스
=========================
Author: Jin
Date: 2025.09.30
Version: 1.0

Description:
2023년 MEPS 보험 데이터의 로딩부터 분석까지 전체 파이프라인을 제공하는 서비스입니다.
Insurance2023DataLoader와 Insurance2023DataAnalyzer를 조합하여 사용하며,
보험 도메인에 특화된 데이터 처리 기능을 제공합니다.
"""
import pandas as pd
from typing import Dict, Any

from base.service_base import ServiceBase
from services.insurance.h2023.insurance_2023_data_loader import Insurance2023DataLoader
from services.insurance.h2023.insurance_2023_data_analyzer import Insurance2023DataAnalyzer
from models.insurance_model import InsuranceDataBatch
from config.logging_config import logger


class Insurance2023Service(ServiceBase):
    """2023년 MEPS 보험 데이터 처리 서비스"""
    
    def __init__(self):
        """
        2023년 MEPS 보험 데이터 처리 서비스 초기화
        
        Insurance2023DataLoader와 Insurance2023DataAnalyzer를 조합하여
        2023년 보험 데이터의 전체 처리 파이프라인을 제공합니다.
        """
        loader = Insurance2023DataLoader()
        analyzer = Insurance2023DataAnalyzer()
        super().__init__(
            name="Insurance2023Service",
            loader=loader,
            analyzer=analyzer
        )
    
    def validate_data(self, filepath: str) -> bool:
        """
        MEPS 2023 보험 CSV인지 검증
        
        Args:
            filepath: 검증할 파일 경로
            
        Returns:
            검증 성공 여부
        """
        try:
            df_sample = pd.read_csv(filepath, nrows=0)
            columns = set(df_sample.columns)
            
            # MEPS 2023 필수 컬럼 확인
            required_columns = {'DUID', 'PID'}
            has_required = required_columns.issubset(columns)
            
            # 2023년 특정 컬럼 확인 (연도 식별)
            year_columns = {'AGE23X', 'TOTEXP23', 'INSCOV23', 'DATAYEAR'}
            has_year_indicator = any(col in columns for col in year_columns)
            
            # 2023년 특화: COVID 및 Long COVID 컬럼 확인
            covid_columns = {'COVIDEVER31', 'COVIDEVER53', 'LCEVER31', 'LCEVER53'}
            has_covid_indicator = any(col in columns for col in covid_columns)
            
            # 2023년 특화: 재향군인 컬럼 확인
            has_veteran = 'EVERSERVED' in columns
            
            if has_required and has_year_indicator:
                logger.info(f"[{self.name}] MEPS 2023 보험 데이터로 감지됨")
                if has_covid_indicator:
                    logger.info(f"[{self.name}] COVID/Long COVID 정보 포함")
                if has_veteran:
                    logger.info(f"[{self.name}] 재향군인 정보 포함")
                return True
            
            logger.warning(f"[{self.name}] MEPS 2023 데이터가 아닙니다")
            return False
            
        except Exception as e:
            logger.error(f"[{self.name}] 검증 실패: {e}")
            return False
    
    def load(self, filepath: str) -> InsuranceDataBatch:
        """
        보험 데이터 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            InsuranceDataBatch 객체
        """
        logger.info(f"[{self.name}] 데이터 로딩 시작: {filepath}")
        batch = self.loader.process(filepath)
        logger.info(f"[{self.name}] 데이터 로딩 완료: {batch.size}개 레코드")
        return batch
    
    def analyze(self, data: InsuranceDataBatch) -> Dict[str, Any]:
        """
        보험 데이터 분석
        
        Args:
            data: 분석할 InsuranceDataBatch
            
        Returns:
            분석 결과 딕셔너리
        """
        logger.info(f"[{self.name}] 데이터 분석 시작")
        
        # 1. 데이터 로드
        self.analyzer.load_insurance_data(data)
        
        # 2. 기본 통계 분석
        logger.info(f"[{self.name}] 기본 통계 분석 중...")
        basic_stats = self.analyzer.basic_statistics() 
        
        # 3. 상관관계 분석
        logger.info(f"[{self.name}] 상관관계 분석 중...")
        correlation = self.analyzer.correlation_analysis()
        
        # 4. 이상치 탐지
        logger.info(f"[{self.name}] 이상치 탐지 중...")
        outliers = self.analyzer.outlier_detection()
        
        # 5. 보험 도메인 특화 분석
        logger.info(f"[{self.name}] 도메인 특화 분석 중...")
        domain_analysis = self.analyzer.domain_specific_analysis() 
        
        # 6. 결과 통합
        results = {
            'service': self.name,
            'data_year': 2023,
            'data_size': data.size,
            'basic_statistics': basic_stats,
            'correlation_analysis': correlation,
            'outlier_detection': outliers,
            'domain_analysis': domain_analysis,
            'summary': self._create_summary(domain_analysis)
        }
        
        logger.info(f"[{self.name}] 데이터 분석 완료")
        return results
    
    def _create_summary(self, domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 결과 요약 생성
        
        Args:
            domain_analysis: 도메인 분석 결과
            
        Returns:
            요약 정보 딕셔너리
        """
        summary = {
            'analysis_completed': True,
            'analysis_categories': list(domain_analysis.keys()),
            'year': 2023
        }
        
        # 보험 가입률 요약
        if 'insurance_coverage_analysis' in domain_analysis:
            coverage = domain_analysis['insurance_coverage_analysis']
            if 'coverage_statistics' in coverage:
                stats = coverage['coverage_statistics']
                summary['coverage_rate'] = stats.get('coverage_rate', 0)
                summary['total_population'] = stats.get('total_population', 0)
        
        # 의료비 지출 요약
        if 'expenditure_analysis' in domain_analysis:
            exp = domain_analysis['expenditure_analysis']
            if 'overall_statistics' in exp:
                stats = exp['overall_statistics']
                summary['avg_expenditure'] = stats.get('mean', 0)
                summary['median_expenditure'] = stats.get('median', 0)
        
        # 보험 유형 개수
        if 'insurance_type_comparison' in domain_analysis:
            summary['insurance_types_count'] = len(domain_analysis['insurance_type_comparison'])
        
        # COVID 백신 접종률 (2023)
        if 'covid_vaccination_analysis' in domain_analysis:
            vacc = domain_analysis['covid_vaccination_analysis']
            if 'vaccination_statistics' in vacc:
                stats = vacc['vaccination_statistics']
                summary['covid_vaccination_rate'] = stats.get('vaccination_rate', 0)
        
        # Long COVID 경험률
        if 'long_covid_analysis' in domain_analysis:
            lc = domain_analysis['long_covid_analysis']
            if 'long_covid_statistics' in lc:
                stats = lc['long_covid_statistics']
                summary['long_covid_rate'] = stats.get('experience_rate', 0)
        
        # 재향군인 비율
        if 'veteran_analysis' in domain_analysis:
            vet = domain_analysis['veteran_analysis']
            if 'veteran_statistics' in vet:
                stats = vet['veteran_statistics']
                summary['veteran_rate'] = stats.get('veteran_rate', 0)
        
        return summary
    
    def process_and_analyze(self, filepath: str) -> Dict[str, Any]:
        """
        데이터 로드 및 분석을 한번에 수행
        
        Args:
            filepath: 처리할 파일 경로
            
        Returns:
            분석 결과 딕셔너리
            
        Raises:
            ValueError: 유효한 MEPS 2023 데이터가 아닌 경우
        """
        logger.info(f"[{self.name}] 전체 프로세스 시작: {filepath}")
        
        # 1. 검증
        if not self.validate_data(filepath):
            raise ValueError("유효한 MEPS 2023 데이터가 아닙니다")
        
        # 2. 로드
        batch = self.load(filepath)
        
        # 3. 분석
        results = self.analyze(batch)
        
        logger.info(f"[{self.name}] 전체 프로세스 완료")
        return results
    
    def get_loader_summary(self) -> Dict[str, Any]:
        """
        로더 요약 정보
        
        Returns:
            로더 요약 정보 딕셔너리
        """
        return self.loader.get_insurance_summary() # type: ignore