"""
H2022 MEPS 문서 모델
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
MEPS HC-243 2022 Full Year Consolidated Data File의 문서 구조를 표현하는 모델입니다.
실제 PDF에서 추출한 필드들을 기반으로 구성되었습니다.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class H2022Document:
    """H2022 MEPS 문서 모델"""
    
    # === 문서 메타데이터 ===
    document_id: str  # 문서 고유 ID
    page_number: int  # 페이지 번호
    total_pages: int  # 총 페이지 수
    content: str  # 페이지 내용
    
    # === 문서 정보 ===
    document_type: str = "MEPS_HC243_2022"  # 문서 타입
    panel: Optional[str] = None  # Panel 번호 (26, 27 등)
    round_info: Optional[str] = None  # Round 정보
    
    # === 섹션 정보 ===
    section: Optional[str] = None  # 섹션 (A, B, C 등)
    subsection: Optional[str] = None  # 하위 섹션
    section_title: Optional[str] = None  # 섹션 제목
    
    # === 변수 정보 (해당 페이지에 포함된 변수들) ===
    variables: List[str] = field(default_factory=list)  # 페이지에 언급된 변수명들
    
    # === 카테고리별 필드 ===
    category: Optional[str] = None  # 카테고리 (Demographics, Health Status, Insurance 등)
    
    # === 추가 메타데이터 ===
    has_table: bool = False  # 테이블 포함 여부
    
    # === 시간 정보 ===
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리로 변환
        
        Returns:
            모델 데이터를 포함한 딕셔너리
        """
        return {
            'document_id': self.document_id,
            'page_number': self.page_number,
            'total_pages': self.total_pages,
            'content': self.content,
            'document_type': self.document_type,
            'panel': self.panel,
            'round_info': self.round_info,
            'section': self.section,
            'subsection': self.subsection,
            'section_title': self.section_title,
            'variables': self.variables,
            'category': self.category,
            'has_table': self.has_table,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'H2022Document':
        """
        딕셔너리에서 모델 생성
        
        Args:
            data: 모델 데이터를 포함한 딕셔너리
            
        Returns:
            H2022Document 인스턴스
        """
        # datetime 변환
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@dataclass
class H2022Variable:
    """H2022 MEPS 변수 정보 모델"""
    
    # === 변수 기본 정보 ===
    variable_name: str  # 변수명 (예: DUPERSID, AGE22X)
    description: str  # 변수 설명
    data_type: str  # 데이터 타입 (NUM, CHAR)
    format: Optional[str] = None  # 포맷 (길이)
    
    # === 변수 위치 ===
    start_position: Optional[int] = None  # 시작 위치
    end_position: Optional[int] = None  # 종료 위치
    
    # === 변수 카테고리 ===
    category: Optional[str] = None  # 카테고리 (Demographics, Health, Insurance 등)
    
    # === 코드 값 ===
    code_values: Dict[str, str] = field(default_factory=dict)  # 코드 값과 설명
    
    # === 추가 정보 ===
    is_edited: bool = False  # 편집된 변수인지 (X로 끝나는 변수)
    round_specific: bool = False  # 라운드별 변수인지
    reserved_codes: Dict[str, str] = field(default_factory=dict)  # 예약 코드 (-1, -7 등)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리로 변환
        
        Returns:
            변수 데이터를 포함한 딕셔너리
        """
        return {
            'variable_name': self.variable_name,
            'description': self.description,
            'data_type': self.data_type,
            'format': self.format,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'category': self.category,
            'code_values': self.code_values,
            'is_edited': self.is_edited,
            'round_specific': self.round_specific,
            'reserved_codes': self.reserved_codes
        }


@dataclass
class H2022Section:
    """H2022 문서 섹션 모델"""
    
    # === 섹션 정보 ===
    section_id: str  # 섹션 ID (A, B, C 등)
    section_name: str  # 섹션 이름
    section_title: str  # 섹션 제목
    
    # === 페이지 정보 ===
    start_page: int  # 시작 페이지
    end_page: Optional[int] = None  # 종료 페이지
    
    # === 내용 ===
    description: Optional[str] = None  # 섹션 설명
    subsections: List[str] = field(default_factory=list)  # 하위 섹션 리스트
    
    # === 포함된 변수 ===
    variables: List[str] = field(default_factory=list)  # 섹션에 포함된 변수들
    
    def to_dict(self) -> Dict[str, Any]:
        """
        딕셔너리로 변환
        
        Returns:
            섹션 데이터를 포함한 딕셔너리
        """
        return {
            'section_id': self.section_id,
            'section_name': self.section_name,
            'section_title': self.section_title,
            'start_page': self.start_page,
            'end_page': self.end_page,
            'description': self.description,
            'subsections': self.subsections,
            'variables': self.variables
        }


# === 주요 카테고리 상수 ===
class H2022Category:
    """H2022 문서 카테고리 정의"""
    DATA_USE_AGREEMENT = "Data Use Agreement"
    BACKGROUND = "Background"
    TECHNICAL_INFO = "Technical and Programming Information"
    IDENTIFIERS = "Unique Person Identifiers"
    GEOGRAPHIC = "Geographic Variables"
    DEMOGRAPHICS = "Demographic Variables"
    INCOME = "Income and Tax Filing Variables"
    HEALTH_CONDITIONS = "Priority Condition and COVID Variables"
    EMPLOYMENT = "Employment Variables"
    INSURANCE = "Health Insurance Variables"
    UTILIZATION = "Utilization, Expenditure, and Source of Payment Variables"
    WEIGHTS = "Weight and Variance Estimation Variables"
    PERSON_STATUS = "Person Status Variables"
    ROUND_SPECIFIC = "Round-Specific Variables"

