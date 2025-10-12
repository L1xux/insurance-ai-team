"""
H2022 전처리기
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
H2022 MEPS 문서 전용 전처리기입니다.
테이블, 변수 정의, 코드 값 등 MEPS 문서의 특수한 구조를 고려하여 청킹합니다.
"""
import re
from typing import List, Dict, Any, Optional

from base.rag.preprocessor_base import PreprocessorBase, Chunk
from config.logging_config import logger


class H2022Preprocessor(PreprocessorBase):
    """H2022 MEPS 문서 전용 전처리기"""
    
    # MEPS 문서 특수 패턴
    TABLE_PATTERN = re.compile(r'Table\s+\d+', re.IGNORECASE)
    SECTION_PATTERN = re.compile(r'Section\s+[A-Z]', re.IGNORECASE)
    VARIABLE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9_]{3,15})\b')
    CODE_VALUE_PATTERN = re.compile(r'-?\d+\s+(Inapplicable|Refused|Don\'t Know|Cannot Be Computed)', re.IGNORECASE)
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        name: Optional[str] = None
    ):
        """
        H2022 전처리기 초기화
        
        Args:
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 겹침 크기
            name: 전처리기 이름
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            name=name or "H2022Preprocessor"
        )
        
       
        logger.info(
            f"[{self.name}] H2022 전처리기 초기화 완료 "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap}, "
      
        )
    
    def preprocess(self, text: str) -> str:
        """
        텍스트 전처리 (MEPS 문서 특화)
        
        Args:
            text: 전처리할 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not text:
            return ""
        
        # 1. 과도한 공백 정리 (하지만 테이블 구조는 유지)
        # 연속된 공백을 하나로 (단, 줄바꿈은 유지)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 각 라인의 공백 정리
            cleaned_line = re.sub(r'[ \t]+', ' ', line)
            cleaned_lines.append(cleaned_line.strip())
        
        text = '\n'.join(cleaned_lines)
        
        # 2. 연속된 빈 줄을 최대 2개로 제한
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. 특수 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # 4. 페이지 번호 패턴 제거 (예: "C-14 MEPS HC 243")
        text = re.sub(r'[A-Z]-\d+\s+MEPS\s+HC\s+\d+', '', text)
        
        return text.strip()
    
    def split_text(self, text: str) -> List[str]:
        """
        MEPS 문서 구조를 고려한 텍스트 청킹
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 텍스트 청크 리스트
        """
        if not text:
            return []
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # 일반 청킹
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # 문장 경계에서 자르기 (가능한 경우)
            if end_pos < len(text):
                # 마지막 마침표 찾기
                chunk_text = text[current_pos:end_pos]
                last_period = chunk_text.rfind('. ')
                last_newline = chunk_text.rfind('\n\n')
                
                # 마침표나 단락이 있으면 그곳에서 자르기
                if last_period > self.chunk_size * 0.7:
                    end_pos = current_pos + last_period + 2
                elif last_newline > self.chunk_size * 0.7:
                    end_pos = current_pos + last_newline + 2
            
            # 청크 생성
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text and len(chunk_text) > 50:  # 너무 짧은 청크 제외
                chunks.append(chunk_text)
            
            # 다음 위치 (오버랩 적용)
            current_pos = end_pos - self.chunk_overlap
            
            # 무한 루프 방지
            if current_pos <= end_pos - self.chunk_size:
                current_pos = end_pos
        
        logger.info(f"[{self.name}] 청킹 완료: {len(chunks)}개 청크 생성")
        
        return chunks
    
    def create_chunks_with_metadata(
        self,
        text: str,
        page_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        메타데이터를 포함한 청크 생성 (H2022 특화)
        
        Args:
            text: 처리할 텍스트
            page_metadata: 페이지 메타데이터
            
        Returns:
            청크 객체 리스트
        """
        # 전처리
        cleaned_text = self.preprocess(text)
        
        # 청킹
        text_chunks = self.split_text(cleaned_text)
        
        # 청크 객체 생성
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            # 메타데이터 복사 및 확장
            chunk_metadata = page_metadata.copy()
            chunk_metadata['chunk_index'] = idx
            chunk_metadata['chunk_size'] = len(chunk_text)
            chunk_metadata['total_chunks'] = len(text_chunks)
            
            # 청크 특성 분석
            chunk_metadata['has_table'] = bool(self.TABLE_PATTERN.search(chunk_text))
            chunk_metadata['has_code_values'] = bool(self.CODE_VALUE_PATTERN.search(chunk_text))
            
            # 청크에 포함된 변수 추출
            variables = self.VARIABLE_PATTERN.findall(chunk_text)
            chunk_metadata['chunk_variables'] = list(set(variables))[:10]  # 최대 10개
            
            chunk = Chunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"h2022_p{page_metadata.get('page_number', 0)}_c{idx}"
            )
            chunks.append(chunk)
        
        self.processed_count += 1
        logger.info(
            f"[{self.name}] 페이지 {page_metadata.get('page_number')} 처리 완료: "
            f"{len(chunks)}개 청크"
        )
        
        return chunks
