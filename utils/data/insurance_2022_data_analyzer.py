"""
보험 데이터 분석기 (2022)
========================
Author: Jin
Date: 2025.09.30
Version: 1.0

Description:
MEPS 2022년 보험 데이터에 특화된 분석 기능을 제공합니다.
- Loader에서 생성한 파생 컬럼(insurance_coverage_label, insurance_type, age_group, income_level 등)을 사용
- 가능한 경우 가중치(_weight = PERWT22F) 지표를 함께 제공
"""
from typing import Dict, Any, Optional
import pandas as pd

from base.agent.data_analyzer_base import DataAnalyzerBase
from models.insurance_model import InsuranceDataBatch
from utils.data.insurance_2022_data_loader import Insurance2022DataLoader
from config.logging_config import logger


class Insurance2022DataAnalyzer(DataAnalyzerBase):
    """2022 보험 데이터 전용 분석기"""

    def __init__(self):
        """
        2022년 보험 데이터 분석기 초기화
        
        DataAnalyzerBase를 상속받아 2022년 보험 데이터 전용 분석기를 생성합니다.
        Loader의 매핑 상수들을 참조하여 정확한 데이터 해석을 제공합니다.
        """
        super().__init__("Insurance2022DataAnalyzer")
        self.insurance_batch: Optional[InsuranceDataBatch] = None

        # Loader의 매핑 상수 참조
        self.INSURC_DETAILED_MAP = Insurance2022DataLoader.INSURC_DETAILED_MAP
        self.HEALTH_STATUS_MAP = Insurance2022DataLoader.HEALTH_STATUS_MAP
        self.SEX_MAP = Insurance2022DataLoader.SEX_MAP
        self.POVERTY_MAP = Insurance2022DataLoader.POVERTY_MAP

        # 고용 상태 맵 
        self.EMPLOYMENT_MAP = {
            1: "Employed",
            2: "Has Job to Return",
            34: "Not Employed",
        }

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

        results: Dict[str, Any] = {}
        try:
            results["insurance_coverage"] = self._analyze_insurance_coverage()
            results["expenditure_patterns"] = self._analyze_expenditure_patterns()
            results["insurance_comparison"] = self._compare_insurance_types()
            results["demographic_disparities"] = self._analyze_demographic_disparities()
            results["financial_burden"] = self._analyze_financial_burden()
            results["health_insurance_access"] = self._analyze_health_access()
            results["covid_impact"] = self._analyze_covid_impact()
            results["vulnerable_populations"] = self._analyze_vulnerable_populations()

            self.analysis_results["insurance_domain_analysis"] = results
            logger.info(f"[{self.name}] 보험 도메인 분석 완료")

        except Exception as e:
            logger.error(f"[{self.name}] 분석 중 오류 발생: {str(e)}")
            raise

        return results

    def _analyze_insurance_coverage(self) -> Dict[str, Any]:
        """
        보험 가입 현황 종합 분석
        
        Returns:
            보험 가입 현황 분석 결과
        """
        res: Dict[str, Any] = {"overall": {}, "by_type": {}, "by_age": {}, "by_income": {}}

        w = self.df["_weight"] if "_weight" in self.df.columns else None

        # 전체 가입 현황
        if "insurance_coverage_label" in self.df.columns:
            insured = (self.df["insurance_coverage_label"] != "").astype(int)
            total = len(self.df)

            res["overall"] = {
                "total_population": int(total),
                "insured_count": int(insured.sum()),
                "uninsured_count": int((1 - insured).sum()),
                "coverage_rate": float(insured.mean() * 100.0),
            }

            if w is not None:
                # 가중 커버리지율
                m = ~insured.isna() & ~w.isna()
                rate_w = float((insured[m] * w[m]).sum() / w[m].sum() * 100.0) if m.sum() else float("nan")
                res["overall"]["coverage_rate_weighted"] = rate_w

        # 상세 보험 유형 분포
        if "INSURC22" in self.df.columns:
            dist = self.df["INSURC22"].value_counts().sort_index()
            res["by_type"]["detailed"] = {
                self.INSURC_DETAILED_MAP.get(int(k), f"Type {int(k)}"): int(v) for k, v in dist.items()
            }

        # 연령대별
        if "age_group" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            g = self.df.groupby("age_group")
            by_age = {}
            for k, sub in g:
                ins = (sub["insurance_coverage_label"] != "").astype(int)
                e = {
                    "total": int(len(sub)),
                    "insured": int(ins.sum()),
                    "coverage_rate": float(ins.mean() * 100.0),
                }
                if "_weight" in sub.columns:
                    ww = sub["_weight"]
                    m = ~ins.isna() & ~ww.isna()
                    e["coverage_rate_weighted"] = float((ins[m] * ww[m]).sum() / ww[m].sum() * 100.0) if m.sum() else float("nan")
                by_age[str(k)] = e
            res["by_age"] = by_age

        # 소득 수준별
        if "income_level" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            g = self.df.groupby("income_level")
            by_inc = {}
            for k, sub in g:
                ins = (sub["insurance_coverage_label"] != "").astype(int)
                e = {
                    "total": int(len(sub)),
                    "insured": int(ins.sum()),
                    "coverage_rate": float(ins.mean() * 100.0),
                }
                if "_weight" in sub.columns:
                    ww = sub["_weight"]
                    m = ~ins.isna() & ~ww.isna()
                    e["coverage_rate_weighted"] = float((ins[m] * ww[m]).sum() / ww[m].sum() * 100.0) if m.sum() else float("nan")
                by_inc[str(k)] = e
            res["by_income"] = by_inc

        return res

    def _analyze_expenditure_patterns(self) -> Dict[str, Any]:
        """
        의료비 지출 패턴 상세 분석
        
        Returns:
            의료비 지출 패턴 분석 결과
        """
        res: Dict[str, Any] = {"total_expenditure": {}, "out_of_pocket": {}, "by_payer": {}, "by_service": {}, "high_cost_cases": {}}

        # 총 의료비
        if "TOTEXP22" in self.df.columns:
            x = self.df["TOTEXP22"].astype(float)
            res["total_expenditure"] = {
                "count": int(len(x)),
                "mean": float(x.mean()),
                "median": float(x.median()),
                "std": float(x.std()),
                "min": float(x.min()),
                "max": float(x.max()),
                "sum": float(x.sum()),
                "percentiles": {
                    "10th": float(x.quantile(0.10)),
                    "25th": float(x.quantile(0.25)),
                    "50th": float(x.quantile(0.50)),
                    "75th": float(x.quantile(0.75)),
                    "90th": float(x.quantile(0.90)),
                    "95th": float(x.quantile(0.95)),
                    "99th": float(x.quantile(0.99)),
                },
            }
            if "expenditure_range" in self.df.columns:
                res["total_expenditure"]["distribution"] = {
                    str(k): int(v) for k, v in self.df["expenditure_range"].value_counts().sort_index().items()
                }

        # 본인부담
        if "TOTSLF22" in self.df.columns:
            s = self.df["TOTSLF22"].astype(float)
            out = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "sum": float(s.sum()),
                "zero_oop_count": int((s == 0).sum()),
                "zero_oop_rate": float((s == 0).mean() * 100.0),
            }
            if "oop_ratio" in self.df.columns:
                r = self.df["oop_ratio"].astype(float)
                out["ratio_stats"] = {"mean": float(r.mean()), "median": float(r.median())}
            res["out_of_pocket"] = out

        # 지불원별
        payer_cols = {
            "Medicare": "TOTMCR22",
            "Medicaid": "TOTMCD22",
            "Private": "TOTPRV22",
            "VA": "TOTVA22",
            "TRICARE": "TOTTRI22",
            "Other_Federal": "TOTOFD22",
            "State_Local": "TOTSTL22",
            "Workers_Comp": "TOTWCP22",
            "Other": "TOTOSR22",
        }
        for name, col in payer_cols.items():
            if col in self.df.columns:
                v = self.df[col].astype(float)
                res["by_payer"][name] = {"total": float(v.sum()), "mean": float(v.mean()), "recipients": int((v > 0).sum())}

        # 서비스 유형별
        service_cols = {
            "Office_Based": "OBVEXP22",
            "Outpatient": "OPTEXP22",
            "Emergency": "ERTEXP22",
            "Inpatient": "IPTEXP22",
            "Prescription": "RXEXP22",
            "Dental": "DVTEXP22",
            "Home_Health": "HHTEXP22",
            "Vision": "VISEXP22",
            "Other_Medical": "OTHEXP22",
        }
        for name, col in service_cols.items():
            if col in self.df.columns:
                v = self.df[col].astype(float)
                res["by_service"][name] = {
                    "total": float(v.sum()),
                    "mean": float(v.mean()),
                    "users": int((v > 0).sum()),
                    "usage_rate": float((v > 0).mean() * 100.0),
                }

        # 고액 의료비 (상위 5%)
        if "TOTEXP22" in self.df.columns:
            thr = float(self.df["TOTEXP22"].quantile(0.95))
            hi = self.df[self.df["TOTEXP22"] >= thr]
            res["high_cost_cases"] = {
                "threshold": thr,
                "count": int(len(hi)),
                "percentage": float(len(hi) / len(self.df) * 100.0) if len(self.df) else 0.0,
                "total_expenditure": float(hi["TOTEXP22"].sum()),
                "share_of_total": float(hi["TOTEXP22"].sum() / self.df["TOTEXP22"].sum() * 100.0)
                if self.df["TOTEXP22"].sum() > 0
                else 0.0,
            }
            if "insurance_type" in hi.columns:
                res["high_cost_cases"]["insurance_distribution"] = {k: int(v) for k, v in hi["insurance_type"].value_counts().items()}

        return res

    def _compare_insurance_types(self) -> Dict[str, Any]:
        """
        보험 유형별 상세 비교 분석
        
        Returns:
            보험 유형별 비교 분석 결과
        """
        out: Dict[str, Any] = {}
        if "insurance_type" not in self.df.columns:
            return out

        total = len(self.df)
        for ins_type in self.df["insurance_type"].unique():
            group = self.df[self.df["insurance_type"] == ins_type]
            prof = {"population": {"count": int(len(group)), "percentage": float(len(group) / total * 100.0)}}

            # 인구통계
            if "AGE22X" in group.columns:
                prof["demographics"] = {
                    "mean_age": float(group["AGE22X"].mean()),
                    "median_age": float(group["AGE22X"].median()),
                }
            else:
                prof["demographics"] = {}

            if "sex_label" in group.columns:
                prof["demographics"]["sex_distribution"] = {k: int(v) for k, v in group["sex_label"].value_counts().items()}

            # 의료비
            if "TOTEXP22" in group.columns:
                prof["expenditure"] = {
                    "mean": float(group["TOTEXP22"].mean()),
                    "median": float(group["TOTEXP22"].median()),
                    "total": float(group["TOTEXP22"].sum()),
                }

            # 본인부담
            if "TOTSLF22" in group.columns:
                prof.setdefault("out_of_pocket", {})
                prof["out_of_pocket"].update(
                    {"mean": float(group["TOTSLF22"].mean()), "median": float(group["TOTSLF22"].median())}
                )
            if "oop_ratio" in group.columns:
                prof.setdefault("out_of_pocket", {})
                prof["out_of_pocket"]["mean_ratio"] = float(group["oop_ratio"].mean())

            # 건강 상태
            if "health_status_label" in group.columns:
                prof["health_status"] = {k: int(v) for k, v in group["health_status_label"].value_counts().items()}

            # 소득 수준
            if "income_level" in group.columns:
                prof["income_distribution"] = {k: int(v) for k, v in group["income_level"].value_counts().items()}

            out[str(ins_type)] = prof

        return out

    def _analyze_demographic_disparities(self) -> Dict[str, Any]:
        """
        인구통계학적 보험 격차
        
        Returns:
            인구통계학적 격차 분석 결과
        """
        res: Dict[str, Any] = {"age_disparities": {}, "sex_disparities": {}, "income_disparities": {}, "employment_disparities": {}}

        # 연령
        if "age_group" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            g = self.df.groupby("age_group")
            for k, sub in g:
                insured = (sub["insurance_coverage_label"] != "").astype(int)
                entry = {
                    "coverage_rate": float(insured.mean() * 100.0),
                    "uninsured_rate": float((1 - insured).mean() * 100.0),
                }
                if "TOTEXP22" in sub.columns:
                    entry["avg_expenditure"] = float(sub["TOTEXP22"].mean())
                res["age_disparities"][str(k)] = entry

        # 성별
        if "SEX" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            g = self.df.groupby("SEX")
            for k, sub in g:
                insured = (sub["insurance_coverage_label"] != "").astype(int)
                name = self.SEX_MAP.get(int(k), str(k))
                entry = {
                    "coverage_rate": float(insured.mean() * 100.0),
                    "uninsured_rate": float((1 - insured).mean() * 100.0),
                }
                if "TOTEXP22" in sub.columns:
                    entry["avg_expenditure"] = float(sub["TOTEXP22"].mean())
                res["sex_disparities"][name] = entry

        # 소득
        if "POVCAT22" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            g = self.df.groupby("POVCAT22")
            for k, sub in g:
                insured = (sub["insurance_coverage_label"] != "").astype(int)
                name = self.POVERTY_MAP.get(int(k), str(k))
                entry = {
                    "coverage_rate": float(insured.mean() * 100.0),
                    "uninsured_rate": float((1 - insured).mean() * 100.0),
                }
                if "TOTEXP22" in sub.columns:
                    entry["avg_expenditure"] = float(sub["TOTEXP22"].mean())
                res["income_disparities"][name] = entry

        # 고용
        emp_col = None
        for c in ["EMPST53", "EMPST42", "EMPST31"]:
            if c in self.df.columns:
                emp_col = c
                break
        if emp_col and "insurance_coverage_label" in self.df.columns:
            g = self.df.groupby(emp_col)
            by_emp = {}
            for k, sub in g:
                insured = (sub["insurance_coverage_label"] != "").astype(int)
                label = self.EMPLOYMENT_MAP.get(int(k), str(k)) if pd.notna(k) else "NA"
                by_emp[label] = {
                    "coverage_rate": float(insured.mean() * 100.0),
                    "uninsured_rate": float((1 - insured).mean() * 100.0),
                }
            res["employment_disparities"] = by_emp

        return res

    def _analyze_financial_burden(self) -> Dict[str, Any]:
        """
        의료비 재정 부담
        
        Returns:
            재정 부담 분석 결과
        """
        res: Dict[str, Any] = {"burden_metrics": {}, "catastrophic_spending": {}, "by_income": {}}

        # 본인부담 비율
        if "oop_ratio" in self.df.columns:
            r = self.df["oop_ratio"].astype(float)
            res["burden_metrics"] = {
                "mean_oop_ratio": float(r.mean()),
                "median_oop_ratio": float(r.median()),
                "high_burden_rate": float((r > 0.2).mean() * 100.0),
                "very_high_burden_rate": float((r > 0.5).mean() * 100.0),
            }

        # 파국적 의료비 (소득 대비)
        if "TOTSLF22" in self.df.columns and "FAMINC22" in self.df.columns:
            slf = self.df["TOTSLF22"].astype(float)
            inc = self.df["FAMINC22"].astype(float)
            ratio = slf / inc.replace({0.0: float("nan")})
            ratio = ratio.fillna(0.0)

            res["catastrophic_spending"] = {
                "over_10pct_income": int((ratio > 0.10).sum()),
                "over_10pct_rate": float((ratio > 0.10).mean() * 100.0),
                "over_20pct_income": int((ratio > 0.20).sum()),
                "over_20pct_rate": float((ratio > 0.20).mean() * 100.0),
                "over_40pct_income": int((ratio > 0.40).sum()),
                "over_40pct_rate": float((ratio > 0.40).mean() * 100.0),
            }

        # 소득 수준별 부담
        if "income_level" in self.df.columns and "oop_ratio" in self.df.columns:
            g = self.df.groupby("income_level")["oop_ratio"].agg(["mean", "median", "count"])
            res["by_income"] = {
                str(idx): {"mean_oop_ratio": float(row["mean"]), "median_oop_ratio": float(row["median"]), "count": int(row["count"])}
                for idx, row in g.iterrows()
            }

        return res

    def _analyze_health_access(self) -> Dict[str, Any]:
        """
        건강 상태와 보험 접근성
        
        Returns:
            건강 상태와 보험 접근성 분석 결과
        """
        res: Dict[str, Any] = {
            "health_coverage_relationship": {},
            "utilization_by_health": {},
            "expenditure_by_health": {},
        }

        if "health_status" not in self.df.columns:
            return res

        # 건강 상태별 가입률
        if "insurance_coverage_label" in self.df.columns:
            g = self.df.groupby("health_status")
            tmp = {}
            for k, sub in g:
                insured = (sub["insurance_coverage_label"] != "").astype(int)
                tmp[self.HEALTH_STATUS_MAP.get(int(k), str(k)) if k == k else "NA"] = {
                    "total": int(len(sub)),
                    "uninsured_rate": float((1 - insured).mean() * 100.0),
                    "coverage_rate": float(insured.mean() * 100.0),
                }
            res["health_coverage_relationship"] = tmp

        # 코드가 있으면 사용
        util_cols = {
            "office_visits": "OBTOTV22",
            "er_visits": "ERTOT22",
            "hospitalizations": "IPDIS22",
            "prescription_fills": "RXTOT22",
        }
        for name, col in util_cols.items():
            if col in self.df.columns:
                g = self.df.groupby("health_status")[col].agg(["mean", "median"])
                res["utilization_by_health"][name] = {
                    self.HEALTH_STATUS_MAP.get(int(idx), str(idx)): {"mean": float(row["mean"]), "median": float(row["median"])}
                    for idx, row in g.iterrows()
                }

        # 지출
        if "TOTEXP22" in self.df.columns:
            g = self.df.groupby("health_status")["TOTEXP22"].agg(["mean", "median"])
            res["expenditure_by_health"] = {
                self.HEALTH_STATUS_MAP.get(int(idx), str(idx)): {"mean": float(row["mean"]), "median": float(row["median"])}
                for idx, row in g.iterrows()
            }

        return res

    def _analyze_covid_impact(self) -> Dict[str, Any]:
        """
        COVID-19 영향
        
        Returns:
            COVID-19 영향 분석 결과
        """
        res: Dict[str, Any] = {"vaccination": {}, "vaccination_by_insurance": {}, "vaccination_disparities": {}}

        if "covid_vaccinated" in self.df.columns:
            v = (self.df["covid_vaccinated"] == 1).astype(int)
            res["vaccination"]["overall"] = {
                "vaccinated_count": int(v.sum()),
                "unvaccinated_count": int((1 - v).sum()),
                "vaccination_rate": float(v.mean() * 100.0),
            }

            # 보험 유형별
            if "insurance_type" in self.df.columns:
                g = self.df.groupby("insurance_type")["covid_vaccinated"].agg(["sum", "count", "mean"])
                res["vaccination_by_insurance"] = {
                    str(idx): {"vaccinated": int(row["sum"]), "total": int(row["count"]), "rate": float(row["mean"] * 100.0)}
                    for idx, row in g.iterrows()
                }

            # 연령대별
            if "age_group" in self.df.columns:
                g = self.df.groupby("age_group")["covid_vaccinated"].mean() * 100.0
                res["vaccination_disparities"]["by_age"] = {str(k): float(v) for k, v in g.items()}

            # 소득별
            if "income_level" in self.df.columns:
                g = self.df.groupby("income_level")["covid_vaccinated"].mean() * 100.0
                res["vaccination_disparities"]["by_income"] = {str(k): float(v) for k, v in g.items()}

        return res

    def _analyze_vulnerable_populations(self) -> Dict[str, Any]:
        """
        취약 계층 보험 접근성
        
        Returns:
            취약 계층 보험 접근성 분석 결과
        """
        res: Dict[str, Any] = {"low_income_uninsured": {}, "young_adults": {}, "chronic_conditions": {}, "high_need_low_coverage": {}}

        # 저소득 무보험
        if "POVCAT22" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            low = self.df[self.df["POVCAT22"].isin([1, 2])]
            if len(low) > 0:
                uninsured = (low["insurance_coverage_label"] == "").sum()
                res["low_income_uninsured"] = {
                    "count": int(uninsured),
                    "rate": float(uninsured / len(low) * 100.0),
                    "total_low_income": int(len(low)),
                }

        # 청년층 (19-25)
        if "AGE22X" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            ya = self.df[(self.df["AGE22X"] >= 19) & (self.df["AGE22X"] <= 25)]
            if len(ya) > 0:
                uninsured = (ya["insurance_coverage_label"] == "").sum()
                res["young_adults"] = {
                    "total": int(len(ya)),
                    "uninsured": int(uninsured),
                    "uninsured_rate": float(uninsured / len(ya) * 100.0),
                }

        # 만성질환 (존재하는 진단코드만 사용)
        chronic_cols = [c for c in ["HIBPDX", "DIABDX", "CHDDX", "ASTHDX"] if c in self.df.columns]
        if chronic_cols and "insurance_coverage_label" in self.df.columns:
            has_chronic = (self.df[chronic_cols] == 1).any(axis=1)
            tot = int(has_chronic.sum())
            if tot > 0:
                uninsured_chronic = int(((self.df["insurance_coverage_label"] == "") & has_chronic).sum())
                res["chronic_conditions"] = {
                    "total_with_chronic": tot,
                    "uninsured_with_chronic": uninsured_chronic,
                    "uninsured_rate": float(uninsured_chronic / tot * 100.0),
                }

        # 고의료필요도(건강상태 Fair/Poor) & 저보장
        if "health_status" in self.df.columns and "insurance_coverage_label" in self.df.columns:
            poor = self.df[self.df["health_status"].isin([4, 5])]
            if len(poor) > 0:
                unins = int((poor["insurance_coverage_label"] == "").sum())
                res["high_need_low_coverage"] = {
                    "poor_health_count": int(len(poor)),
                    "uninsured_poor_health": unins,
                    "uninsured_rate": float(unins / len(poor) * 100.0),
                }

        return res
