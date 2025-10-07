"""
보험 데이터 분석 서비스
=========================
Author: Jin
Date: 2025.09.30
Version: 1.2

Description:
2023년 MEPS 보험 데이터에 특화된 분석 기능을 제공하는 서비스입니다.
"""
from typing import Dict, Any, Optional

from base.data_analyzer_base import DataAnalyzerBase
from models.insurance_model import InsuranceDataBatch
from services.insurance.h2023.insurance_2023_data_loader import Insurance2023DataLoader
from config.logging_config import logger


class Insurance2023DataAnalyzer(DataAnalyzerBase):
    """2023 보험 데이터 전용 분석기"""
    
    def __init__(self):
        """
        2023년 보험 데이터 분석기 초기화
        
        DataAnalyzerBase를 상속받아 2023년 보험 데이터 전용 분석기를 생성합니다.
        Loader의 매핑 상수들을 참조하여 정확한 데이터 해석을 제공합니다.
        """
        super().__init__("Insurance2023DataAnalyzer")
        self.insurance_batch: Optional[InsuranceDataBatch] = None
        
        # Loader의 매핑 상수 참조
        self.INSURANCE_TYPE_MAP = Insurance2023DataLoader.INSURANCE_TYPE_MAP
        self.INSURC_DETAILED_MAP = Insurance2023DataLoader.INSURC_DETAILED_MAP
        self.HEALTH_STATUS_MAP = Insurance2023DataLoader.HEALTH_STATUS_MAP
        self.SEX_MAP = Insurance2023DataLoader.SEX_MAP
        self.EMPLOYMENT_MAP = Insurance2023DataLoader.EMPLOYMENT_MAP
        self.REGION_MAP = Insurance2023DataLoader.REGION_MAP
        self.RACE_MAP = Insurance2023DataLoader.RACE_MAP
        self.POVERTY_MAP = Insurance2023DataLoader.POVERTY_MAP
    
    def load_insurance_data(self, insurance_batch: InsuranceDataBatch) -> None:
        """
        보험 데이터 배치 로드
        
        Args:
            insurance_batch: 로드할 보험 데이터 배치
            
        Raises:
            ValueError: 유효하지 않은 InsuranceDataBatch인 경우
        """
        if insurance_batch is None or insurance_batch.size == 0:
            raise ValueError("유효하지 않은 InsuranceDataBatch입니다.")
        
        self.insurance_batch = insurance_batch
        df = insurance_batch.to_dataframe()
        self.load_data(df)
        
        logger.info(f"[{self.name}] 보험 데이터 로드: {insurance_batch.size}개 레코드")
    
    def domain_specific_analysis(self) -> Dict[str, Any]:
        """보험 도메인 특화 분석"""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        results = {}
        
        try:
            # 1. 보험 가입 현황 종합 분석
            results['insurance_coverage'] = self._analyze_insurance_coverage()
            
            # 2. 의료비 지출 패턴 분석
            results['expenditure_patterns'] = self._analyze_expenditure_patterns()
            
            # 3. 보험 유형별 상세 비교
            results['insurance_comparison'] = self._compare_insurance_types()
            
            # 4. 인구통계학적 보험 격차 분석
            results['demographic_disparities'] = self._analyze_demographic_disparities()
            
            # 5. 본인부담금 부담 분석
            results['financial_burden'] = self._analyze_financial_burden()
            
            # 6. 건강 상태와 보험 접근성
            results['health_insurance_access'] = self._analyze_health_access()
            
            # 7. COVID-19 영향 분석
            results['covid_impact'] = self._analyze_covid_impact()
            
            # 8. Long COVID 의료비 부담
            results['long_covid_burden'] = self._analyze_long_covid_burden()
            
            # 9. 재향군인 보험 현황
            results['veteran_coverage'] = self._analyze_veteran_coverage()
            
            # 10. 취약 계층 분석
            results['vulnerable_populations'] = self._analyze_vulnerable_populations()
            
            self.analysis_results['insurance_domain_analysis'] = results
            logger.info(f"[{self.name}] 보험 도메인 분석 완료")
            
        except Exception as e:
            logger.error(f"[{self.name}] 분석 중 오류 발생: {str(e)}")
            raise
            
        return results
    
    def _analyze_insurance_coverage(self) -> Dict[str, Any]:
        """보험 가입 현황 종합 분석"""
        results = {
            'overall': {},
            'by_type': {},
            'by_age': {},
            'by_income': {},
            'by_region': {},
            'trends': {}
        }
        
        # 전체 가입 현황
        if 'INSCOV23' in self.df.columns:
            inscov_dist = self.df['INSCOV23'].value_counts()
            total = len(self.df)
            
            results['overall'] = {
                'total_population': total,
                'any_private': int(inscov_dist.get(1, 0)),
                'public_only': int(inscov_dist.get(2, 0)),
                'uninsured': int(inscov_dist.get(3, 0)),
                'coverage_rate': float((total - inscov_dist.get(3, 0)) / total * 100) if total > 0 else 0,
                'private_rate': float(inscov_dist.get(1, 0) / total * 100) if total > 0 else 0,
                'public_rate': float(inscov_dist.get(2, 0) / total * 100) if total > 0 else 0,
                'uninsured_rate': float(inscov_dist.get(3, 0) / total * 100) if total > 0 else 0
            }
        
        # 상세 보험 유형별
        if 'INSURC23' in self.df.columns:
            insurc_dist = self.df['INSURC23'].value_counts().sort_index()
            results['by_type']['detailed'] = {
                self.INSURC_DETAILED_MAP.get(k, f'Type {k}'): int(v)
                for k, v in insurc_dist.items()
            }
        
        # 연령대별 가입률
        if 'age_group' in self.df.columns and 'INSCOV23' in self.df.columns:
            age_coverage = self.df.groupby('age_group').agg({
                'INSCOV23': lambda x: {
                    'total': len(x),
                    'insured': (x != 3).sum(),
                    'uninsured': (x == 3).sum(),
                    'coverage_rate': (x != 3).mean() * 100
                }
            })
            
            results['by_age'] = {
                str(idx): val['INSCOV23']
                for idx, val in age_coverage.iterrows()
            }
        
        # 소득 수준별 가입률
        if 'income_level' in self.df.columns and 'INSCOV23' in self.df.columns:
            income_coverage = self.df.groupby('income_level').agg({
                'INSCOV23': lambda x: {
                    'total': len(x),
                    'insured': (x != 3).sum(),
                    'coverage_rate': (x != 3).mean() * 100,
                    'private_rate': (x == 1).mean() * 100,
                    'public_rate': (x == 2).mean() * 100
                }
            })
            
            results['by_income'] = {
                str(idx): val['INSCOV23']
                for idx, val in income_coverage.iterrows()
            }
        
        # 지역별 가입률
        if 'REGION23' in self.df.columns and 'INSCOV23' in self.df.columns:
            region_coverage = self.df.groupby('REGION23').agg({
                'INSCOV23': lambda x: {
                    'total': len(x),
                    'coverage_rate': (x != 3).mean() * 100
                }
            })
            
            results['by_region'] = {
                self.REGION_MAP.get(idx, str(idx)): val['INSCOV23']
                for idx, val in region_coverage.iterrows()
            }
        
        return results
    
    def _analyze_expenditure_patterns(self) -> Dict[str, Any]:
        """의료비 지출 패턴 상세 분석"""
        results = {
            'total_expenditure': {},
            'out_of_pocket': {},
            'by_payer': {},
            'by_service': {},
            'high_cost_cases': {}
        }
        
        # 총 의료비 지출 통계
        if 'TOTEXP23' in self.df.columns:
            exp = self.df['TOTEXP23']
            
            results['total_expenditure'] = {
                'count': int(len(exp)),
                'mean': float(exp.mean()),
                'median': float(exp.median()),
                'std': float(exp.std()),
                'min': float(exp.min()),
                'max': float(exp.max()),
                'sum': float(exp.sum()),
                'percentiles': {
                    '10th': float(exp.quantile(0.10)),
                    '25th': float(exp.quantile(0.25)),
                    '50th': float(exp.quantile(0.50)),
                    '75th': float(exp.quantile(0.75)),
                    '90th': float(exp.quantile(0.90)),
                    '95th': float(exp.quantile(0.95)),
                    '99th': float(exp.quantile(0.99))
                }
            }
            
            # 지출 범위별 분포
            if 'expenditure_range' in self.df.columns:
                results['total_expenditure']['distribution'] = \
                    self.df['expenditure_range'].value_counts().sort_index().to_dict()
        
        # 본인부담금
        if 'TOTSLF23' in self.df.columns:
            oop = self.df['TOTSLF23']
            
            results['out_of_pocket'] = {
                'mean': float(oop.mean()),
                'median': float(oop.median()),
                'sum': float(oop.sum()),
                'zero_oop_count': int((oop == 0).sum()),
                'zero_oop_rate': float((oop == 0).mean() * 100)
            }
            
            # 본인부담 비율
            if 'oop_ratio' in self.df.columns:
                oop_ratio = self.df['oop_ratio']
                results['out_of_pocket']['ratio_stats'] = {
                    'mean': float(oop_ratio.mean()),
                    'median': float(oop_ratio.median())
                }
        
        # 지불원별 지출
        payer_columns = {
            'Medicare': 'TOTMCR23',
            'Medicaid': 'TOTMCD23',
            'Private': 'TOTPRV23',
            'VA': 'TOTVA23',
            'TRICARE': 'TOTTRI23',
            'Other_Federal': 'TOTOFD23',
            'State_Local': 'TOTSTL23',
            'Workers_Comp': 'TOTWCP23',
            'Other': 'TOTOSR23'
        }
        
        for payer_name, col in payer_columns.items():
            if col in self.df.columns:
                payer_exp = self.df[col]
                results['by_payer'][payer_name] = {
                    'total': float(payer_exp.sum()),
                    'mean': float(payer_exp.mean()),
                    'recipients': int((payer_exp > 0).sum())
                }
        
        # 서비스 유형별 지출
        service_columns = {
            'Office_Based': 'OBVEXP23',
            'Outpatient': 'OPTEXP23',
            'Emergency': 'ERTEXP23',
            'Inpatient': 'IPTEXP23',
            'Prescription': 'RXEXP23',
            'Dental': 'DVTEXP23',
            'Home_Health': 'HHTEXP23',
            'Vision': 'VISEXP23',
            'Other_Medical': 'OTHEXP23'
        }
        
        for service_name, col in service_columns.items():
            if col in self.df.columns:
                service_exp = self.df[col]
                results['by_service'][service_name] = {
                    'total': float(service_exp.sum()),
                    'mean': float(service_exp.mean()),
                    'users': int((service_exp > 0).sum()),
                    'usage_rate': float((service_exp > 0).mean() * 100)
                }
        
        # 고액 의료비 케이스 분석
        if 'TOTEXP23' in self.df.columns:
            high_cost_threshold = self.df['TOTEXP23'].quantile(0.95)
            high_cost = self.df[self.df['TOTEXP23'] >= high_cost_threshold]
            
            results['high_cost_cases'] = {
                'threshold': float(high_cost_threshold),
                'count': len(high_cost),
                'percentage': float(len(high_cost) / len(self.df) * 100),
                'total_expenditure': float(high_cost['TOTEXP23'].sum()),
                'share_of_total': float(
                    high_cost['TOTEXP23'].sum() / self.df['TOTEXP23'].sum() * 100
                ) if self.df['TOTEXP23'].sum() > 0 else 0
            }
            
            # 고액 환자의 보험 유형 분포
            if 'insurance_type' in high_cost.columns:
                results['high_cost_cases']['insurance_distribution'] = \
                    high_cost['insurance_type'].value_counts().to_dict()
        
        return results
    
    def _compare_insurance_types(self) -> Dict[str, Any]:
        """보험 유형별 상세 비교 분석"""
        results = {}
        
        if 'insurance_type' not in self.df.columns:
            return results
        
        for ins_type in self.df['insurance_type'].unique():
            group = self.df[self.df['insurance_type'] == ins_type]
            
            type_profile = {
                'population': {
                    'count': len(group),
                    'percentage': float(len(group) / len(self.df) * 100)
                }
            }
            
            # 인구통계
            if 'AGE23X' in group.columns:
                type_profile['demographics'] = {
                    'mean_age': float(group['AGE23X'].mean()),
                    'median_age': float(group['AGE23X'].median())
                }
            
            if 'sex_label' in group.columns:
                sex_dist = group['sex_label'].value_counts()
                type_profile['demographics']['sex_distribution'] = sex_dist.to_dict()
            
            # 의료비 지출
            if 'TOTEXP23' in group.columns:
                type_profile['expenditure'] = {
                    'mean': float(group['TOTEXP23'].mean()),
                    'median': float(group['TOTEXP23'].median()),
                    'total': float(group['TOTEXP23'].sum())
                }
            
            # 본인부담금
            if 'TOTSLF23' in group.columns:
                type_profile['out_of_pocket'] = {
                    'mean': float(group['TOTSLF23'].mean()),
                    'median': float(group['TOTSLF23'].median())
                }
                
                if 'oop_ratio' in group.columns:
                    type_profile['out_of_pocket']['mean_ratio'] = \
                        float(group['oop_ratio'].mean())
            
            # 건강 상태
            if 'health_status_label' in group.columns:
                health_dist = group['health_status_label'].value_counts()
                type_profile['health_status'] = health_dist.to_dict()
            
            # 고용 상태
            if 'EMPST53' in group.columns:
                emp_dist = group['EMPST53'].map(
                    lambda x: self.EMPLOYMENT_MAP.get(x, 'Other')
                ).value_counts()
                type_profile['employment'] = emp_dist.to_dict()
            
            # 소득 수준
            if 'income_level' in group.columns:
                type_profile['income_distribution'] = \
                    group['income_level'].value_counts().to_dict()
            
            results[str(ins_type)] = type_profile
        
        return results
    
    def _analyze_demographic_disparities(self) -> Dict[str, Any]:
        """인구통계학적 보험 격차 분석"""
        results = {
            'age_disparities': {},
            'sex_disparities': {},
            'race_disparities': {},
            'income_disparities': {},
            'employment_disparities': {}
        }
        
        # 연령별 격차
        if 'age_group' in self.df.columns and 'INSCOV23' in self.df.columns:
            age_analysis = self.df.groupby('age_group').agg({
                'INSCOV23': lambda x: {
                    'uninsured_rate': (x == 3).mean() * 100,
                    'private_rate': (x == 1).mean() * 100,
                    'public_rate': (x == 2).mean() * 100
                },
                'TOTEXP23': 'mean' if 'TOTEXP23' in self.df.columns else lambda x: 0
            })
            
            results['age_disparities'] = {
                str(idx): {
                    **row['INSCOV23'],
                    'avg_expenditure': float(row.get('TOTEXP23', 0))
                }
                for idx, row in age_analysis.iterrows()
            }
        
        # 성별 격차
        if 'SEX' in self.df.columns and 'INSCOV23' in self.df.columns:
            sex_analysis = self.df.groupby('SEX').agg({
                'INSCOV23': lambda x: {
                    'uninsured_rate': (x == 3).mean() * 100,
                    'coverage_rate': (x != 3).mean() * 100
                },
                'TOTEXP23': 'mean' if 'TOTEXP23' in self.df.columns else lambda x: 0
            })
            
            results['sex_disparities'] = {
                self.SEX_MAP.get(idx, str(idx)): {
                    **row['INSCOV23'],
                    'avg_expenditure': float(row.get('TOTEXP23', 0))
                }
                for idx, row in sex_analysis.iterrows()
            }
        
        # 인종/민족별 격차
        if 'RACEV1X' in self.df.columns and 'INSCOV23' in self.df.columns:
            race_analysis = self.df.groupby('RACEV1X').agg({
                'INSCOV23': lambda x: {
                    'uninsured_rate': (x == 3).mean() * 100,
                    'coverage_rate': (x != 3).mean() * 100
                }
            })
            
            results['race_disparities'] = {
                self.RACE_MAP.get(idx, str(idx)): row['INSCOV23']
                for idx, row in race_analysis.iterrows()
            }
        
        # 소득별 격차
        if 'POVCAT23' in self.df.columns and 'INSCOV23' in self.df.columns:
            income_analysis = self.df.groupby('POVCAT23').agg({
                'INSCOV23': lambda x: {
                    'uninsured_rate': (x == 3).mean() * 100,
                    'private_rate': (x == 1).mean() * 100,
                    'public_rate': (x == 2).mean() * 100
                },
                'TOTEXP23': 'mean' if 'TOTEXP23' in self.df.columns else lambda x: 0
            })
            
            results['income_disparities'] = {
                self.POVERTY_MAP.get(idx, str(idx)): {
                    **row['INSCOV23'],
                    'avg_expenditure': float(row.get('TOTEXP23', 0))
                }
                for idx, row in income_analysis.iterrows()
            }
        
        # 고용 상태별 격차
        if 'EMPST53' in self.df.columns and 'INSCOV23' in self.df.columns:
            emp_analysis = self.df.groupby('EMPST53').agg({
                'INSCOV23': lambda x: {
                    'uninsured_rate': (x == 3).mean() * 100,
                    'private_rate': (x == 1).mean() * 100
                }
            })
            
            results['employment_disparities'] = {
                self.EMPLOYMENT_MAP.get(idx, str(idx)): row['INSCOV23']
                for idx, row in emp_analysis.iterrows()
            }
        
        return results
    
    def _analyze_financial_burden(self) -> Dict[str, Any]:
        """의료비 재정 부담 분석"""
        results = {
            'burden_metrics': {},
            'catastrophic_spending': {},
            'by_income': {}
        }
        
        # 본인부담 부담률
        if 'oop_ratio' in self.df.columns:
            oop_burden = self.df['oop_ratio']
            
            results['burden_metrics'] = {
                'mean_oop_ratio': float(oop_burden.mean()),
                'median_oop_ratio': float(oop_burden.median()),
                'high_burden_rate': float((oop_burden > 0.2).mean() * 100),
                'very_high_burden_rate': float((oop_burden > 0.5).mean() * 100)
            }
        
        # 파국적 의료비 지출 (소득 대비)
        if 'TOTSLF23' in self.df.columns and 'FAMINC23' in self.df.columns:
            income_ratio = self.df.apply(
                lambda row: row['TOTSLF23'] / row['FAMINC23'] 
                if row['FAMINC23'] > 0 else 0,
                axis=1
            )
            
            results['catastrophic_spending'] = {
                'over_10pct_income': int((income_ratio > 0.10).sum()),
                'over_10pct_rate': float((income_ratio > 0.10).mean() * 100),
                'over_20pct_income': int((income_ratio > 0.20).sum()),
                'over_20pct_rate': float((income_ratio > 0.20).mean() * 100),
                'over_40pct_income': int((income_ratio > 0.40).sum()),
                'over_40pct_rate': float((income_ratio > 0.40).mean() * 100)
            }
        
        # 소득 수준별 부담
        if 'income_level' in self.df.columns and 'oop_ratio' in self.df.columns:
            income_burden = self.df.groupby('income_level')['oop_ratio'].agg([
                'mean', 'median', 'count'
            ])
            
            results['by_income'] = {
                str(idx): {
                    'mean_oop_ratio': float(row['mean']),
                    'median_oop_ratio': float(row['median']),
                    'count': int(row['count'])
                }
                for idx, row in income_burden.iterrows()
            }
        
        return results
    
    def _analyze_health_access(self) -> Dict[str, Any]:
        """건강 상태와 보험 접근성 분석"""
        results = {
            'health_coverage_relationship': {},
            'utilization_by_health': {},
            'expenditure_by_health': {}
        }
        
        if 'health_status' not in self.df.columns:
            return results
        
        # 건강 상태별 보험 가입률
        if 'INSCOV23' in self.df.columns:
            health_coverage = self.df.groupby('health_status').agg({
                'INSCOV23': lambda x: {
                    'total': len(x),
                    'uninsured_rate': (x == 3).mean() * 100,
                    'private_rate': (x == 1).mean() * 100,
                    'public_rate': (x == 2).mean() * 100
                }
            })
            
            results['health_coverage_relationship'] = {
                self.HEALTH_STATUS_MAP.get(idx, str(idx)): row['INSCOV23']
                for idx, row in health_coverage.iterrows()
            }
        
        # 건강 상태별 의료 이용
        utilization_cols = {
            'office_visits': 'OBTOTV23',
            'er_visits': 'ERTOT23',
            'hospitalizations': 'IPDIS23',
            'prescription_fills': 'RXTOT23'
        }
        
        for util_name, col in utilization_cols.items():
            if col in self.df.columns:
                health_util = self.df.groupby('health_status')[col].agg(['mean', 'median'])
                results['utilization_by_health'][util_name] = {
                    self.HEALTH_STATUS_MAP.get(idx, str(idx)): {
                        'mean': float(row['mean']),
                        'median': float(row['median'])
                    }
                    for idx, row in health_util.iterrows()
                }
        
        # 건강 상태별 의료비 지출
        if 'TOTEXP23' in self.df.columns:
            health_exp = self.df.groupby('health_status')['TOTEXP23'].agg(['mean', 'median'])
            results['expenditure_by_health'] = {
                self.HEALTH_STATUS_MAP.get(idx, str(idx)): {
                    'mean': float(row['mean']),
                    'median': float(row['median'])
                }
                for idx, row in health_exp.iterrows()
            }
        
        return results
    
    def _analyze_covid_impact(self) -> Dict[str, Any]:
        """COVID-19 영향 분석"""
        results = {
            'vaccination': {},
            'covid_history': {},
            'vaccination_by_insurance': {},
            'vaccination_disparities': {}
        }
        
        # COVID 백신 접종 현황
        if 'covid_vaccinated' in self.df.columns:
            vacc_rate = (self.df['covid_vaccinated'] == 1).mean() * 100
            
            results['vaccination']['overall'] = {
                'vaccinated_count': int((self.df['covid_vaccinated'] == 1).sum()),
                'unvaccinated_count': int((self.df['covid_vaccinated'] == 0).sum()),
                'vaccination_rate': float(vacc_rate)
            }
            
            # 보험 유형별 백신 접종률
            if 'insurance_type' in self.df.columns:
                vacc_by_ins = self.df.groupby('insurance_type')['covid_vaccinated'].agg([
                    'sum', 'count', 'mean'
                ])
                
                results['vaccination_by_insurance'] = {
                    str(idx): {
                        'vaccinated': int(row['sum']),
                        'total': int(row['count']),
                        'rate': float(row['mean'] * 100)
                    }
                    for idx, row in vacc_by_ins.iterrows()
                }
            
            # 연령대별 백신 접종률
            if 'age_group' in self.df.columns:
                vacc_by_age = self.df.groupby('age_group')['covid_vaccinated'].mean() * 100
                results['vaccination_disparities']['by_age'] = vacc_by_age.to_dict()
            
            # 소득별 백신 접종률
            if 'income_level' in self.df.columns:
                vacc_by_income = self.df.groupby('income_level')['covid_vaccinated'].mean() * 100
                results['vaccination_disparities']['by_income'] = vacc_by_income.to_dict()
        
        # COVID-19 감염 이력
        if 'had_covid' in self.df.columns:
            covid_rate = (self.df['had_covid'] == 1).mean() * 100
            results['covid_history'] = {
                'ever_had_covid': int((self.df['had_covid'] == 1).sum()),
                'rate': float(covid_rate)
            }
        
        return results
    
    def _analyze_long_covid_burden(self) -> Dict[str, Any]:
        """Long COVID 의료비 부담 분석"""
        results = {
            'prevalence': {},
            'expenditure_impact': {},
            'insurance_coverage': {},
            'demographic_patterns': {}
        }
        
        # 전처리에서 이미 생성된 long_covid 사용
        if 'long_covid' not in self.df.columns:
            return results
        
        # Long COVID 유병률
        lc_count = (self.df['long_covid'] == 1).sum()
        lc_rate = (self.df['long_covid'] == 1).mean() * 100
        
        results['prevalence'] = {
            'cases': int(lc_count),
            'rate': float(lc_rate),
            'total_population': len(self.df)
        }
        
        # Long COVID와 의료비 비교
        if 'TOTEXP23' in self.df.columns:
            lc_exp = self.df.groupby('long_covid')['TOTEXP23'].agg(['mean', 'median', 'count'])
            
            results['expenditure_impact'] = {
                'with_long_covid': {
                    'mean': float(lc_exp.loc[1, 'mean']) if 1 in lc_exp.index else 0,
                    'median': float(lc_exp.loc[1, 'median']) if 1 in lc_exp.index else 0,
                    'count': int(lc_exp.loc[1, 'count']) if 1 in lc_exp.index else 0
                },
                'without_long_covid': {
                    'mean': float(lc_exp.loc[0, 'mean']) if 0 in lc_exp.index else 0,
                    'median': float(lc_exp.loc[0, 'median']) if 0 in lc_exp.index else 0,
                    'count': int(lc_exp.loc[0, 'count']) if 0 in lc_exp.index else 0
                }
            }
            
            # 추가 의료비 부담
            if 1 in lc_exp.index and 0 in lc_exp.index:
                results['expenditure_impact']['additional_burden'] = {
                    'mean_difference': float(lc_exp.loc[1, 'mean'] - lc_exp.loc[0, 'mean']),
                    'median_difference': float(lc_exp.loc[1, 'median'] - lc_exp.loc[0, 'median'])
                }
        
        # Long COVID 환자의 보험 가입 현황
        if 'insurance_type' in self.df.columns:
            lc_patients = self.df[self.df['long_covid'] == 1]
            if len(lc_patients) > 0:
                ins_dist = lc_patients['insurance_type'].value_counts()
                results['insurance_coverage'] = ins_dist.to_dict()
        
        # 인구통계학적 패턴
        if 'age_group' in self.df.columns:
            lc_by_age = self.df.groupby('age_group')['long_covid'].mean() * 100
            results['demographic_patterns']['by_age'] = lc_by_age.to_dict()
        
        if 'sex_label' in self.df.columns:
            lc_by_sex = self.df.groupby('sex_label')['long_covid'].mean() * 100
            results['demographic_patterns']['by_sex'] = lc_by_sex.to_dict()
        
        return results
    
    def _analyze_veteran_coverage(self) -> Dict[str, Any]:
        """재향군인 보험 현황 분석"""
        results = {
            'veteran_statistics': {},
            'insurance_comparison': {},
            'va_coverage': {},
            'expenditure_comparison': {}
        }
        
        if 'veteran_status' not in self.df.columns:
            return results
        
        # 재향군인 통계
        vet_count = (self.df['veteran_status'] == 1).sum()
        vet_rate = (self.df['veteran_status'] == 1).mean() * 100
        
        results['veteran_statistics'] = {
            'veteran_count': int(vet_count),
            'non_veteran_count': int((self.df['veteran_status'] == 0).sum()),
            'veteran_rate': float(vet_rate)
        }
        
        # 재향군인 vs 비재향군인 보험 가입 비교
        if 'INSCOV23' in self.df.columns:
            vet_insurance = self.df.groupby('veteran_status').agg({
                'INSCOV23': lambda x: {
                    'uninsured_rate': (x == 3).mean() * 100,
                    'coverage_rate': (x != 3).mean() * 100,
                    'private_rate': (x == 1).mean() * 100,
                    'public_rate': (x == 2).mean() * 100
                }
            })
            
            results['insurance_comparison'] = {
                'veterans': vet_insurance.loc[1, 'INSCOV23'] if 1 in vet_insurance.index else {},
                'non_veterans': vet_insurance.loc[0, 'INSCOV23'] if 0 in vet_insurance.index else {}
            }
        
        # VA 보험 이용 현황
        if 'uses_va' in self.df.columns:
            veterans = self.df[self.df['veteran_status'] == 1]
            if len(veterans) > 0:
                va_users = (veterans['uses_va'] == 1).sum()
                results['va_coverage'] = {
                    'va_users': int(va_users),
                    'va_usage_rate': float((va_users / len(veterans)) * 100) if len(veterans) > 0 else 0
                }
        
        # 의료비 지출 비교
        if 'TOTEXP23' in self.df.columns:
            vet_exp = self.df.groupby('veteran_status')['TOTEXP23'].agg(['mean', 'median'])
            
            results['expenditure_comparison'] = {
                'veterans': {
                    'mean': float(vet_exp.loc[1, 'mean']) if 1 in vet_exp.index else 0,
                    'median': float(vet_exp.loc[1, 'median']) if 1 in vet_exp.index else 0
                },
                'non_veterans': {
                    'mean': float(vet_exp.loc[0, 'mean']) if 0 in vet_exp.index else 0,
                    'median': float(vet_exp.loc[0, 'median']) if 0 in vet_exp.index else 0
                }
            }
        
        return results
    
    def _analyze_vulnerable_populations(self) -> Dict[str, Any]:
        """취약 계층 보험 접근성 분석"""
        results = {
            'low_income_uninsured': {},
            'young_adults': {},
            'chronic_conditions': {},
            'high_need_low_coverage': {}
        }
        
        # 저소득 무보험자
        if 'POVCAT23' in self.df.columns and 'INSCOV23' in self.df.columns:
            low_income = self.df[self.df['POVCAT23'].isin([1, 2])]
            if len(low_income) > 0:
                uninsured = (low_income['INSCOV23'] == 3).sum()
                results['low_income_uninsured'] = {
                    'count': int(uninsured),
                    'rate': float((uninsured / len(low_income)) * 100)
                }
        
        # 청년층 (19-25세) 보험 현황
        if 'AGE23X' in self.df.columns and 'INSCOV23' in self.df.columns:
            young_adults = self.df[(self.df['AGE23X'] >= 19) & (self.df['AGE23X'] <= 25)]
            if len(young_adults) > 0:
                ya_uninsured = (young_adults['INSCOV23'] == 3).sum()
                results['young_adults'] = {
                    'total': len(young_adults),
                    'uninsured': int(ya_uninsured),
                    'uninsured_rate': float((ya_uninsured / len(young_adults)) * 100)
                }
        
        # 만성질환자 무보험 현황
        if 'has_chronic_condition' in self.df.columns and 'INSCOV23' in self.df.columns:
            chronic_patients = self.df[self.df['has_chronic_condition'] == 1]
            if len(chronic_patients) > 0:
                chronic_uninsured = (chronic_patients['INSCOV23'] == 3).sum()
                
                results['chronic_conditions'] = {
                    'total_with_chronic': int((self.df['has_chronic_condition'] == 1).sum()),
                    'uninsured_with_chronic': int(chronic_uninsured),
                    'uninsured_rate': float((chronic_uninsured / len(chronic_patients)) * 100)
                }
        
        # 고의료필요도 저보장 집단
        if 'poor_health' in self.df.columns and 'INSCOV23' in self.df.columns:
            poor_health = self.df[self.df['poor_health'] == 1]
            if len(poor_health) > 0:
                uninsured_poor_health = (poor_health['INSCOV23'] == 3).sum()
                results['high_need_low_coverage'] = {
                    'poor_health_count': len(poor_health),
                    'uninsured_poor_health': int(uninsured_poor_health),
                    'uninsured_rate': float((uninsured_poor_health / len(poor_health)) * 100)
                }
        
        return results