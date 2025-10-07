"""
보험 데이터 로더
=========================
Author: Jin
Date: 2025.09.30
Version: 1.1

Description:
MEPS(Medical Expenditure Panel Survey) 2023년 보험 데이터를 로드하고 전처리합니다.
DataLoaderBase를 상속받아 보험 도메인 특화 기능을 제공합니다.
"""
import pandas as pd
from typing import Dict, Any

from base.data_loader_base import DataLoaderBase
from models.insurance_model import InsuranceDataBatch
from config.logging_config import logger


class Insurance2023DataLoader(DataLoaderBase):
    """2023 보험 데이터 전용 로더 - DataLoaderBase 상속"""
    
    # 2023 MEPS 코드 매핑
    INSURANCE_TYPE_MAP = {
        1: 'Any Private',
        2: 'Public Only', 
        3: 'Uninsured'
    }
    
    INSURC_DETAILED_MAP = {
        1: 'Any Private (0-64)',
        2: 'Public Only (0-64)',
        3: 'Uninsured (0-64)',
        4: 'Medicare Only (65+)',
        5: 'Medicare & Private (65+)',
        6: 'Medicare & Other Public (65+)',
        7: 'Uninsured (65+)',
        8: 'No Medicare but Public/Private (65+)'
    }
    
    HEALTH_STATUS_MAP = {
        1: 'Excellent',
        2: 'Very Good',
        3: 'Good',
        4: 'Fair',
        5: 'Poor'
    }
    
    SEX_MAP = {1: 'Male', 2: 'Female'}
    
    EMPLOYMENT_MAP = {
        1: 'Employed',
        2: 'Has Job to Return',
        34: 'Not Employed'
    }
    
    REGION_MAP = {
        1: 'Northeast',
        2: 'Midwest',
        3: 'South',
        4: 'West'
    }
    
    RACE_MAP = {
        1: 'White',
        2: 'Black',
        3: 'American Indian/Alaska Native',
        4: 'Asian/Native Hawaiian/Pacific Islander',
        6: 'Multiple Races'
    }
    
    POVERTY_MAP = {
        1: 'Poor/Negative (<100% FPL)',
        2: 'Near Poor (100-<125% FPL)',
        3: 'Low Income (125-<200% FPL)',
        4: 'Middle Income (200-<400% FPL)',
        5: 'High Income (≥400% FPL)'
    }
    
    def __init__(self):
        """
        2023년 보험 데이터 로더 초기화
        
        DataLoaderBase를 상속받아 2023년 MEPS 보험 데이터 전용 로더를 생성합니다.
        MEPS 2023 데이터의 주요 컬럼들을 그룹별로 정의하여 체계적인 데이터 처리를 제공합니다.
        """
        super().__init__("Insurance2023DataLoader")
        
        # MEPS 2023 데이터 주요 컬럼 정의
        self.key_columns = {
            'demographic': ['DUID', 'PID', 'AGE23X', 'SEX', 'MARRY23X', 'RACEV1X', 'HISPANX', 'EVERSERVED'],
            'insurance': ['INSCOV23', 'INSURC23', 'UNINS23', 'PRVEV23', 'MCREV23', 'MCDEV23', 'VAEV23'],
            'expenditure': ['TOTEXP23', 'TOTSLF23', 'TOTMCR23', 'TOTMCD23', 'TOTPRV23', 
                          'TOTVA23', 'TOTTRI23', 'TOTOFD23', 'TOTSTL23', 'TOTWCP23', 'TOTOSR23'],
            'service': ['OBVEXP23', 'OPTEXP23', 'ERTEXP23', 'IPTEXP23', 'RXEXP23', 
                       'DVTEXP23', 'HHTEXP23', 'VISEXP23', 'OTHEXP23'],
            'utilization': ['OBTOTV23', 'ERTOT23', 'IPDIS23', 'RXTOT23'],
            'health': ['RTHLTH31', 'RTHLTH42', 'RTHLTH53', 'MNHLTH31', 'MNHLTH42', 'MNHLTH53',
                      'HIBPDX', 'DIABDX_M18', 'CHDDX', 'ASTHDX'],
            'income': ['FAMINC23', 'POVCAT23', 'POVLEV23', 'TTLP23X'],
            'employment': ['EMPST31', 'EMPST42', 'EMPST53'],
            'region': ['REGION23'],
            'covid': ['COVIDEVER31', 'COVIDEVER53', 'LCEVER31', 'LCEVER53', 
                     'COVAXEVR31', 'COVAXEVR42', 'COVAXEVR53']
        }
    
    def process(self, filepath: str) -> InsuranceDataBatch:
        """
        보험 데이터 처리 - 로드 및 전처리
        
        Args:
            filepath: 처리할 CSV 파일 경로
            
        Returns:
            InsuranceDataBatch 객체
            
        Raises:
            ValueError: 데이터 로드 실패 시
            Exception: 처리 중 오류 발생 시
        """
        try:
            # 1. 데이터 로드
            self.load_csv(filepath)
            
            if self.df is None:
                raise ValueError("데이터 로드에 실패했습니다.")
            
            # 2. 전처리
            self._preprocess_insurance_data()
            
            # 3. InsuranceDataBatch 객체 생성
            batch = InsuranceDataBatch.from_dataframe(self.df)
            
            logger.info(f"[{self.name}] InsuranceDataBatch 생성 완료: {batch.size}개 레코드")
            return batch
            
        except Exception as e:
            logger.error(f"[{self.name}] 보험 데이터 처리 실패: {str(e)}")
            raise
    
    def _preprocess_insurance_data(self) -> None:
        """
        보험 데이터 전처리
        
        MEPS 결측 코드 정리, 표본가중치 저장, 보험 유형 분류,
        연령대 분류, 소득 수준 분류, 의료비 지출 처리, 건강 상태 분류,
        인구통계학적 정보 처리 등을 수행합니다.
        """
        if self.df is None:
            return
        
        logger.info(f"[{self.name}] 보험 데이터 전처리 시작...")
        
        # 1. 보험 가입 상태 파생 변수
        self._process_insurance_status()
        
        # 2. 보험 유형 분류
        self._categorize_insurance_type()
        
        # 3. 연령대 분류
        self._categorize_age_groups()
        
        # 4. 소득 수준 분류
        self._categorize_income_levels()
        
        # 5. 의료비 지출 처리
        self._process_expenditure()
        
        # 6. 건강 상태 처리
        self._process_health_status()
        
        # 7. COVID 관련 변수 처리
        self._process_covid_variables()
        
        # 8. 군복무 여부 처리
        self._process_veteran_status()
        
        # 9. 지역 정보 처리
        self._process_region()
        
        # 10. 인종/민족 정보 처리
        self._process_race_ethnicity()
        
        logger.info(f"[{self.name}] 전처리 완료: {self.df.shape}")
    
    def _process_insurance_status(self) -> None:
        """보험 가입 상태 처리"""
        if 'INSCOV23' in self.df.columns:
            self.df['has_insurance'] = (self.df['INSCOV23'] != 3).fillna(False).astype(int)
            self.df['is_uninsured'] = (self.df['INSCOV23'] == 3).fillna(False).astype(int)
            
            # 레이블 매핑
            self.df['insurance_coverage_label'] = self.df['INSCOV23'].map(
                self.INSURANCE_TYPE_MAP
            )
    
    def _categorize_insurance_type(self) -> None:
        """
        보험 유형 분류
        
        Medicare, Medicaid, Private, Uninsured 등의 보험 유형으로 분류합니다.
        """
        if self.df is None:
            return
        
        # 기존 단순 분류 유지
        conditions = [
            (self.df.get('MCREV23', 0) == 1, 'Medicare'),
            (self.df.get('MCDEV23', 0) == 1, 'Medicaid'),
            (self.df.get('PRVEV23', 0) == 1, 'Private'),
            (self.df.get('UNINS23', 0) == 1, 'Uninsured')
        ]
        
        self.df['insurance_type'] = 'Other'
        for condition, label in conditions:
            if isinstance(condition, pd.Series):
                self.df.loc[condition, 'insurance_type'] = label
        
        # 상세 보험 유형 레이블 매핑
        if 'INSURC23' in self.df.columns:
            self.df['insurance_detailed_label'] = self.df['INSURC23'].map(
                self.INSURC_DETAILED_MAP
            )
    
    def _categorize_age_groups(self) -> None:
        """
        연령대 분류
        
        연령을 구간별로 분류하고 아동, 근로연령, 고령자 등의 플래그를 생성합니다.
        """
        if 'AGE23X' in self.df.columns:
            self.df['age_group'] = pd.cut(
                self.df['AGE23X'], 
                bins=[0, 18, 35, 50, 65, 100],
                labels=['0-18', '19-35', '36-50', '51-65', '65+']
            )
            
            # 추가 연령 구분
            self.df['is_child'] = (self.df['AGE23X'] < 18).fillna(False).astype(int)
            self.df['is_working_age'] = (
                (self.df['AGE23X'] >= 18) & (self.df['AGE23X'] < 65)
            ).fillna(False).astype(int)
            self.df['is_senior'] = (self.df['AGE23X'] >= 65).fillna(False).astype(int)
    
    def _categorize_income_levels(self) -> None:
        """
        소득 수준 분류
        
        소득 수준을 분류하고 저소득, 고소득 등의 플래그를 생성합니다.
        """
        if 'POVCAT23' in self.df.columns:
            # 레이블 매핑
            self.df['income_level'] = self.df['POVCAT23'].map(self.POVERTY_MAP)
            
            # 추가 소득 구분
            self.df['is_low_income'] = (self.df['POVCAT23'].isin([1, 2, 3])).fillna(False).astype(int)
            self.df['is_high_income'] = (self.df['POVCAT23'] == 5).fillna(False).astype(int)
    
    def _process_expenditure(self) -> None:
        """의료비 지출 처리"""
        # 총 의료비 지출
        if 'TOTEXP23' in self.df.columns:
            self.df['TOTEXP23'] = self.df['TOTEXP23'].fillna(0)
            self.df['has_expenditure'] = (self.df['TOTEXP23'] > 0).fillna(False).astype(int)
            
            # 지출 범위 분류
            self.df['expenditure_range'] = pd.cut(
                self.df['TOTEXP23'],
                bins=[0, 500, 1000, 2500, 5000, 10000, 25000, 50000, float('inf')],
                labels=['$0-500', '$500-1K', '$1K-2.5K', '$2.5K-5K', 
                       '$5K-10K', '$10K-25K', '$25K-50K', '$50K+']
            )
        
        # 본인부담금
        if 'TOTSLF23' in self.df.columns:
            self.df['TOTSLF23'] = self.df['TOTSLF23'].fillna(0)
            self.df['has_oop'] = (self.df['TOTSLF23'] > 0).fillna(False).astype(int)
        
        # 본인부담비율 계산
        if 'TOTEXP23' in self.df.columns and 'TOTSLF23' in self.df.columns:
            self.df['oop_ratio'] = self.df.apply(
                lambda row: row['TOTSLF23'] / row['TOTEXP23'] 
                if row['TOTEXP23'] > 0 else 0, 
                axis=1
            )
            
            # 높은 본인부담 플래그
            self.df['high_oop_burden'] = (self.df['oop_ratio'] > 0.2).astype(int)
    
    def _process_health_status(self) -> None:
        """건강 상태 처리"""
        # 최신 건강 상태 사용
        health_cols = ['RTHLTH53', 'RTHLTH42', 'RTHLTH31']
        for col in health_cols:
            if col in self.df.columns:
                self.df['health_status'] = self.df[col]
                break
        
        # 건강 상태 레이블 매핑
        if 'health_status' in self.df.columns:
            self.df['health_status_label'] = self.df['health_status'].map(
                self.HEALTH_STATUS_MAP
            )
            
            # 건강 상태 그룹핑
            self.df['poor_health'] = (self.df['health_status'].isin([4, 5])).astype(int)
            self.df['good_health'] = (self.df['health_status'].isin([1, 2])).astype(int)
        
        # 만성질환 플래그
        chronic_conditions = ['HIBPDX', 'DIABDX_M18', 'CHDDX', 'ASTHDX']
        has_chronic = pd.Series(False, index=self.df.index)
        
        for cond in chronic_conditions:
            if cond in self.df.columns:
                has_chronic |= (self.df[cond] == 1)
        
        self.df['has_chronic_condition'] = has_chronic.fillna(False).astype(int)
    
    def _process_covid_variables(self) -> None:
        """COVID 관련 변수 처리"""
        # COVID 백신 접종 여부
        covid_cols = ['COVAXEVR53', 'COVAXEVR42', 'COVAXEVR31']
        for col in covid_cols:
            if col in self.df.columns:
                self.df['covid_vaccinated'] = (self.df[col] == 1).fillna(False).astype(int)
                break
        
        # Long COVID 경험 여부
        lc_cols = ['LCEVER53', 'LCEVER31']
        for col in lc_cols:
            if col in self.df.columns:
                self.df['long_covid'] = (self.df[col] == 1).fillna(False).astype(int)
                break
        
        # COVID 감염 이력
        covid_ever_cols = ['COVIDEVER53', 'COVIDEVER31']
        for col in covid_ever_cols:
            if col in self.df.columns:
                self.df['had_covid'] = (self.df[col] == 1).fillna(False).astype(int)
                break
    
    def _process_veteran_status(self) -> None:
        """군복무 여부 처리"""
        if 'EVERSERVED' in self.df.columns:
            self.df['veteran_status'] = (self.df['EVERSERVED'] == 1).fillna(False).astype(int)
        
        # VA 보험 이용 여부
        if 'VAEV23' in self.df.columns:
            self.df['uses_va'] = (self.df['VAEV23'] == 1).fillna(False).astype(int)
    
    def _process_region(self) -> None:
        """지역 정보 처리"""
        if 'REGION23' in self.df.columns:
            self.df['region_label'] = self.df['REGION23'].map(self.REGION_MAP)
    
    def _process_race_ethnicity(self) -> None:
        """인종/민족 정보 처리"""
        if 'RACEV1X' in self.df.columns:
            self.df['race_label'] = self.df['RACEV1X'].map(self.RACE_MAP)
        
        if 'SEX' in self.df.columns:
            self.df['sex_label'] = self.df['SEX'].map(self.SEX_MAP)
    
    def get_insurance_summary(self) -> Dict[str, Any]:
        """
        보험 데이터 요약 정보
        
        Returns:
            요약 정보 딕셔너리
        """
        if self.df is None:
            return {}
        
        summary = {
            'total_records': len(self.df),
            'insurance_coverage': {},
            'insurance_types': {},
            'expenditure_summary': {},
            'covid_vaccination': {},
            'long_covid': {},
            'veteran_info': {}
        }
        
        # 보험 가입률
        if 'has_insurance' in self.df.columns:
            summary['insurance_coverage'] = {
                'insured': int((self.df['has_insurance'] == 1).sum()),
                'uninsured': int((self.df['has_insurance'] == 0).sum()),
                'coverage_rate': float((self.df['has_insurance'] == 1).mean() * 100)
            }
        
        # 보험 유형 분포
        if 'insurance_type' in self.df.columns:
            summary['insurance_types'] = self.df['insurance_type'].value_counts().to_dict()
        
        # 의료비 지출 요약
        if 'TOTEXP23' in self.df.columns:
            summary['expenditure_summary'] = {
                'mean': float(self.df['TOTEXP23'].mean()),
                'median': float(self.df['TOTEXP23'].median()),
                'total': float(self.df['TOTEXP23'].sum()),
                'max': float(self.df['TOTEXP23'].max())
            }
        
        # COVID 백신 접종률
        if 'covid_vaccinated' in self.df.columns:
            summary['covid_vaccination'] = {
                'vaccinated': int((self.df['covid_vaccinated'] == 1).sum()),
                'vaccination_rate': float((self.df['covid_vaccinated'] == 1).mean() * 100)
            }
        
        # Long COVID 경험률
        if 'long_covid' in self.df.columns:
            summary['long_covid'] = {
                'experienced': int((self.df['long_covid'] == 1).sum()),
                'rate': float((self.df['long_covid'] == 1).mean() * 100)
            }
        
        # 재향군인 정보
        if 'veteran_status' in self.df.columns:
            summary['veteran_info'] = {
                'veterans': int((self.df['veteran_status'] == 1).sum()),
                'rate': float((self.df['veteran_status'] == 1).mean() * 100)
            }
        
        return summary