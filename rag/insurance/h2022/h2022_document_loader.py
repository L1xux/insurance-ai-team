"""
H2022 PDF 문서 로더
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
H2022 MEPS 문서 PDF를 로드하고 구조화된 정보를 추출하는 구현체입니다.
PyPDF2를 사용하여 PDF 파일을 읽고, H2022Document 모델로 변환합니다.
"""
import re
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2

from base.rag.document_loader_base import DocumentLoaderBase, Document
from models.h2022_document_model import (
    H2022Document,
    H2022Category
)
from config.logging_config import logger


class H2022DocumentLoader(DocumentLoaderBase):
    """H2022 PDF 문서 로더"""
    
    # 섹션 패턴
    SECTION_PATTERNS = {
        'A': 'Data Use Agreement',
        'B': 'Background',
        'C': 'Technical and Programming Information'
    }
    
    # 변수 패턴 (대문자로 시작하는 변수명)
    VARIABLE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9_]{3,15})\b')
    
    def __init__(self, name: Optional[str] = None):
        """
        H2022 문서 로더 초기화
        
        Args:
            name: 로더 이름
        """
        super().__init__(name=name or "H2022DocumentLoader")
        logger.info(f"[{self.name}] H2022 PDF 문서 로더 초기화 완료")
    
    def _extract_variables(self, text: str) -> List[str]:
        """
        텍스트에서 변수명 추출
        
        Args:
            text: 추출할 텍스트
            
        Returns:
            발견된 변수명 리스트
        """
        matches = self.VARIABLE_PATTERN.findall(text)
        # 너무 일반적인 단어 제외
        excluded = {'TABLE', 'SECTION', 'PAGE', 'VALUE', 'DEFINITION', 'VARIABLE', 'TYPE'}
        return [m for m in set(matches) if m not in excluded and len(m) > 4]
    
    def _detect_section(self, text: str) -> Optional[str]:
        """
        텍스트에서 섹션 감지
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            섹션 ID 또는 None
        """
        for section_id, section_name in self.SECTION_PATTERNS.items():
            if section_name in text[:500]:  # 페이지 상단에서만 찾기
                return section_id
        return None
    
    def _detect_category(self, text: str, variables: List[str]) -> Optional[str]:
        """
        텍스트와 변수를 기반으로 카테고리 감지
        
        Args:
            text: 분석할 텍스트
            variables: 페이지의 변수들
            
        Returns:
            카테고리 또는 None
        """
        text_lower = text.lower()
        
        # 텍스트 패턴 기반 카테고리 감지
        if 'demographic' in text_lower or 'age' in text_lower or 'gender' in text_lower:
            return H2022Category.DEMOGRAPHICS
        elif 'employment' in text_lower or 'job' in text_lower or 'work' in text_lower:
            return H2022Category.EMPLOYMENT
        elif 'insurance' in text_lower or 'coverage' in text_lower:
            return H2022Category.INSURANCE
        elif 'health status' in text_lower or 'condition' in text_lower:
            return H2022Category.HEALTH_CONDITIONS
        elif 'expenditure' in text_lower or 'utilization' in text_lower:
            return H2022Category.UTILIZATION
        elif 'weight' in text_lower or 'variance' in text_lower:
            return H2022Category.WEIGHTS
        elif 'geographic' in text_lower or 'region' in text_lower:
            return H2022Category.GEOGRAPHIC
        elif 'income' in text_lower or 'poverty' in text_lower:
            return H2022Category.INCOME
        
        # 변수 패턴 기반 카테고리 감지
        if any(v.startswith('AGE') or v.startswith('SEX') or v.startswith('RACE') for v in variables):
            return H2022Category.DEMOGRAPHICS
        elif any(v.startswith('INSCOP') or v.startswith('PSTATS') for v in variables):
            return H2022Category.PERSON_STATUS
        
        return None
    
    def _has_table(self, text: str) -> bool:
        """
        페이지에 테이블이 있는지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            테이블 포함 여부
        """
        return 'Table' in text[:200] or '|' in text or 'Value' in text and 'Definition' in text
    
    def _has_code_values(self, text: str) -> bool:
        """
        페이지에 코드 값 정의가 있는지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            코드 값 포함 여부
        """
        return bool(re.search(r'-?\d+\s+(Inapplicable|Refused|Don\'t Know)', text))
    
    def _is_variable_definition(self, text: str) -> bool:
        """
        변수 정의 페이지인지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            변수 정의 페이지 여부
        """
        return 'Variable' in text and ('Description' in text or 'Format' in text or 'Type' in text)
    
    def load(self, filepath: str) -> List[Document]:
        """
        PDF 파일에서 문서 로드 및 구조화
        
        Args:
            filepath: 로드할 PDF 파일 경로
            
        Returns:
            로드된 문서 리스트
        """
        try:
            
            # 파일 경로 검증
            file_path = Path(filepath)
            if not file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
            if not file_path.suffix.lower() == '.pdf':
                raise ValueError(f"PDF 파일이 아닙니다: {filepath}")
            
            logger.info(f"[{self.name}] PDF 파일 로드 중: {filepath}")
            
            documents = []
            
            # PDF 읽기
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"[{self.name}] 총 {total_pages}페이지 감지")
                
                # 각 페이지별로 텍스트 추출 및 구조화
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # 변수 추출
                        variables = self._extract_variables(text)
                        
                        # 섹션 감지
                        section = self._detect_section(text)
                        
                        # 카테고리 감지
                        category = self._detect_category(text, variables)
                        
                        # H2022Document 모델 생성
                        doc_id = str(uuid.uuid4())
                        h2022_doc = H2022Document(
                            document_id=doc_id,
                            page_number=page_num + 1,
                            total_pages=total_pages,
                            content=text.strip(),
                            document_type="MEPS_HC243_2022",
                            year=2022,
                            section=section,
                            variables=variables,
                            category=category,
                            has_table=self._has_table(text),
                            has_code_values=self._has_code_values(text),
                            is_variable_definition=self._is_variable_definition(text)
                        )
                        
                        # Document 객체로 변환 (메타데이터에 모델 정보 포함)
                        metadata = h2022_doc.to_dict()
                        metadata['source'] = str(file_path)
                        metadata['filename'] = file_path.name
                        
                        doc = Document(
                            content=text.strip(),
                            metadata=metadata
                        )
                        documents.append(doc)
                        
                        if (page_num + 1) % 50 == 0:
                            logger.debug(
                                f"[{self.name}] 진행 상황: {page_num + 1}/{total_pages} 페이지"
                            )
            
            self.loaded_count += 1
            logger.info(
                f"[{self.name}] PDF 로드 완료: {len(documents)}개 문서 "
                f"(총 {sum(len(d.content) for d in documents):,} 문자)"
            )
            
            # 통계 정보 로깅
            total_vars = sum(len(d.metadata.get('variables', [])) for d in documents)
            logger.info(f"[{self.name}] 추출된 고유 변수 수: {total_vars}개")
            
            return documents
            
        except Exception as e:
            logger.error(f"[{self.name}] PDF 로드 실패: {str(e)}")
            raise
    
    def load_batch(self, filepaths: List[str]) -> List[Document]:
        """
        여러 PDF 파일에서 문서를 일괄 로드
        
        Args:
            filepaths: 로드할 PDF 파일 경로 리스트
            
        Returns:
            로드된 모든 문서 리스트
        """
        try:
            all_documents = []
            
            for filepath in filepaths:
                try:
                    documents = self.load(filepath)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"[{self.name}] {filepath} 로드 실패: {str(e)}")
                    continue
            
            logger.info(
                f"[{self.name}] 배치 로드 완료: "
                f"{len(filepaths)}개 파일, {len(all_documents)}개 문서"
            )
            
            return all_documents
            
        except Exception as e:
            logger.error(f"[{self.name}] 배치 로드 실패: {str(e)}")
            raise
    