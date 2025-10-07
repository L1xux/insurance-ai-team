"""
보험 데이터 로더
=============================
Author: Jin
Date: 2025.09.30
Version: 1.2

Description:
MEPS 2022년 보험 데이터를 로드/전처리
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

from base.data_loader_base import DataLoaderBase
from models.insurance_model import InsuranceDataBatch
from config.logging_config import logger


class Insurance2022DataLoader(DataLoaderBase):
    """2022 보험 데이터 전용 로더"""

    # 2022 MEPS 코드 매핑
    INSURC_DETAILED_MAP = {
        1: 'Any Private (0-64)',
        2: 'Public Only (0-64)',
        3: 'Uninsured (0-64)',
        4: 'Medicare Only (65+)',
        5: 'Medicare & Private (65+)',
        6: 'Medicare & Other Public (65+)',
        7: 'Uninsured (65+)',
        8: 'No Medicare but Public/Private (65+)',
    }

    HEALTH_STATUS_MAP = {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'}
    SEX_MAP = {1: 'Male', 2: 'Female'}
    POVERTY_MAP = {
        1: 'Poor/Negative (<100% FPL)',
        2: 'Near Poor (100-<125% FPL)',
        3: 'Low Income (125-<200% FPL)',
        4: 'Middle Income (200-<400% FPL)',
        5: 'High Income (≥400% FPL)',
    }

    def __init__(self):
        """
        2022년 보험 데이터 로더 초기화
        
        DataLoaderBase를 상속받아 2022년 MEPS 보험 데이터 전용 로더를 생성합니다.
        MEPS 2022 데이터의 주요 컬럼들을 그룹별로 정의하여 체계적인 데이터 처리를 제공합니다.
        """
        super().__init__("Insurance2022DataLoader")
        self.key_columns: Dict[str, list[str]] = {
            'demographic': ['DUID', 'PID', 'AGE22X', 'SEX', 'MARRY22X', 'RACEV1X', 'HISPANX'],
            'insurance':  ['INSCOV22', 'INSURC22', 'UNINS22', 'PRVEV22', 'MCREV22', 'MCDEV22'],
            'expenditure':['TOTEXP22', 'TOTSLF22', 'TOTMCR22', 'TOTMCD22', 'TOTPRV22'],
            'service':    ['OBVEXP22', 'OPTEXP22', 'ERTEXP22', 'IPTEXP22', 'RXEXP22',
                           'DVTEXP22', 'HHTEXP22', 'VISEXP22', 'OTHEXP22'],
            'utilization':['OBTOTV22', 'ERTOT22', 'IPDIS22', 'RXTOT22'],
            'health':     ['RTHLTH53', 'RTHLTH42', 'RTHLTH31', 'MNHLTH53', 'MNHLTH42', 'MNHLTH31'],
            'income':     ['FAMINC22', 'POVCAT22', 'POVLEV22', 'TTLP22X'],
            'employment': ['EMPST53', 'EMPST42', 'EMPST31'],
            'covid':      ['COVAXEVR53', 'COVAXEVR42', 'COVAXEVR31', 'BOOSTERSHOT42', 'BOOSTERSHOT31'],
            'weight':     ['PERWT22F'],
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
            self.load_csv(filepath)
            if self.df is None:
                raise ValueError("데이터 로드에 실패했습니다.")

            self._preprocess_insurance_data()

            batch = InsuranceDataBatch.from_dataframe(self.df)
            logger.info(f"[{self.name}] InsuranceDataBatch 생성 완료: {batch.size}개 레코드")
            return batch

        except Exception as e:
            logger.error(f"[{self.name}] 보험 데이터 처리 실패: {str(e)}")
            raise

    def _preprocess_insurance_data(self) -> None:
        """
        보험 데이터 전처리 (맵 기반 파생, 유틸 없음)
        
        MEPS 결측 코드 정리, 표본가중치 저장, 보험 유형 분류,
        연령대 분류, 소득 수준 분류, 의료비 지출 처리, 건강 상태 분류,
        COVID 백신 접종 정보 처리, 성별 라벨링 등을 수행합니다.
        """
        if self.df is None:
            return

        logger.info(f"[{self.name}] 보험 데이터 전처리 시작...")

        # MEPS 결측 코드 정리
        for group_cols in self.key_columns.values():
            for c in group_cols:
                if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c]):
                    s = self.df[c]
                    self.df[c] = s.where(~s.isin([-1, -7, -8, -9]), np.nan)

        # 표본가중치 저장, 없으면 1.0
        self.df['_weight'] = self.df['PERWT22F'] if 'PERWT22F' in self.df.columns else 1.0

        # 상세 보험 유형 레이블
        if 'INSURC22' in self.df.columns:
            self.df['insurance_detailed_label'] = self.df['INSURC22'].map(self.INSURC_DETAILED_MAP)

        # 간단 보험 유형(출처 신호 기반)
        self._categorize_insurance_type()

        # 연령대 분류
        if 'AGE22X' in self.df.columns:
            age = self.df['AGE22X']
            
            self.df['age_group'] = pd.cut(
                self.df['AGE22X'],
                bins=[0, 18, 35, 50, 65, 200],
                labels=['0-18', '19-35', '36-50', '51-65', '65+'],
                right=False
            )
            self.df['is_child'] = (age < 18).fillna(False).astype(int)
            self.df['is_working_age'] = ((age >= 18) & (age < 65)).fillna(False).astype(int)
            self.df['is_senior'] = (age >= 65).fillna(False).astype(int)

        # 소득 수준 분류
        if 'POVCAT22' in self.df.columns:
            self.df['income_level'] = self.df['POVCAT22'].map(self.POVERTY_MAP)
            self.df['is_low_income'] = self.df['POVCAT22'].isin([1, 2, 3]).fillna(False).astype(int)
            self.df['is_high_income'] = (self.df['POVCAT22'] == 5).fillna(False).astype(int)

        # 의료비 지출
        if 'TOTEXP22' in self.df.columns:
            self.df['TOTEXP22'] = self.df['TOTEXP22'].fillna(0)
            self.df['has_expenditure'] = (self.df['TOTEXP22'] > 0).astype(int)
            self.df['expenditure_range'] = pd.cut(
                self.df['TOTEXP22'],
                bins=[0, 500, 1000, 2500, 5000, 10000, 25000, 50000, np.inf],
                labels=['$0-500', '$500-1K', '$1K-2.5K', '$2.5K-5K', '$5K-10K',
                        '$10K-25K', '$25K-50K', '$50K+'],
                right=False
            )

        if 'TOTSLF22' in self.df.columns:
            self.df['TOTSLF22'] = self.df['TOTSLF22'].fillna(0)

        # 벡터화된 본인부담비율
        if {'TOTEXP22', 'TOTSLF22'}.issubset(self.df.columns):
            denom = self.df['TOTEXP22'].replace({0: np.nan})
            self.df['oop_ratio'] = (self.df['TOTSLF22'] / denom).fillna(0.0)
            self.df['high_oop_burden'] = (self.df['oop_ratio'] > 0.2).astype(int)

        # 건강 상태: 최신 라운드 우선
        for col in ['RTHLTH53', 'RTHLTH42', 'RTHLTH31']:
            if col in self.df.columns:
                self.df['health_status'] = self.df[col]
                break
        if 'health_status' in self.df.columns:
            self.df['health_status_label'] = self.df['health_status'].map(self.HEALTH_STATUS_MAP)
            self.df['poor_health'] = self.df['health_status'].isin([4, 5]).fillna(False).astype(int)
            self.df['good_health'] = self.df['health_status'].isin([1, 2]).fillna(False).astype(int)

        # COVID 백신 접종: 최신 라운드 우선 선택
        for col in ['COVAXEVR53', 'COVAXEVR42', 'COVAXEVR31']:
            if col in self.df.columns:
                self.df['covid_vaccinated'] = (self.df[col] == 1).fillna(False).astype(int)
                break
        
        # 성별 라벨
        if 'SEX' in self.df.columns:
            self.df['sex_label'] = self.df['SEX'].map(self.SEX_MAP)

        logger.info(f"[{self.name}] 전처리 완료: {self.df.shape}")


    def _categorize_insurance_type(self) -> None:
        """
        보험 유형 분류(존재 컬럼만 사용, 맵 기반 파생과 병행)
        
        Medicare, Medicaid, Private, Uninsured 등의 보험 유형으로 분류합니다.
        """
        if self.df is None:
            return

        self.df['insurance_type'] = 'Other'
        if 'MCREV22' in self.df.columns:
            self.df.loc[self.df['MCREV22'] == 1, 'insurance_type'] = 'Medicare'
        if 'MCDEV22' in self.df.columns:
            self.df.loc[self.df['MCDEV22'] == 1, 'insurance_type'] = 'Medicaid'
        if 'PRVEV22' in self.df.columns:
            self.df.loc[self.df['PRVEV22'] == 1, 'insurance_type'] = 'Private'
        if 'UNINS22' in self.df.columns:
            self.df.loc[self.df['UNINS22'] == 1, 'insurance_type'] = 'Uninsured'

    def get_insurance_summary(self) -> Dict[str, Any]:
        """
        보험 데이터 요약 (맵 기반 파생 활용)
        
        Returns:
            보험 가입률, 유형별 분포, 의료비 지출 요약, COVID 백신 접종률을 포함한 딕셔너리
        """
        if self.df is None:
            return {}

        summary: Dict[str, Any] = {
            'total_records': int(len(self.df)),
            'insurance_coverage': {},
            'insurance_types': {},
            'expenditure_summary': {},
            'covid_vaccination': {},
        }

        if 'has_insurance' in self.df.columns:
            insured = (self.df['has_insurance'] == 1)
            summary['insurance_coverage'] = {
                'insured': int(insured.sum()),
                'uninsured': int((~insured).sum()),
                'coverage_rate': float(insured.mean() * 100.0),
            }

        if 'insurance_type' in self.df.columns:
            vc = self.df['insurance_type'].value_counts(dropna=False)
            summary['insurance_types'] = {str(k): int(v) for k, v in vc.items()}

        if 'TOTEXP22' in self.df.columns:
            x = self.df['TOTEXP22'].astype(float)
            summary['expenditure_summary'] = {
                'mean': float(x.mean()),
                'median': float(x.median()),
                'total': float(x.sum()),
                'max': float(x.max()),
            }

        if 'covid_vaccinated' in self.df.columns:
            v = (self.df['covid_vaccinated'] == 1)
            summary['covid_vaccination'] = {
                'vaccinated': int(v.sum()),
                'vaccination_rate': float(v.mean() * 100.0),
            }

        return summary
