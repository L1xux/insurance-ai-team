"""
기본 Preprocessor 구현체
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
텍스트를 청크로 분할하고 전처리하는 기본 구현체입니다.
문장 기반 청킹과 기본적인 텍스트 정제 기능을 제공합니다.
"""
import re
from typing import List, Optional

from base.rag.preprocessor_base import PreprocessorBase
from config.logging_config import logger


class Processor(PreprocessorBase):
    """기본 전처리기 구현체"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        name: Optional[str] = None
    ):
        """
        전처리기 초기화
        
        Args:
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 겹침 크기
            separator: 텍스트 분리 구분자
            name: 전처리기 이름
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            name=name
        )
        
        self.separator = separator
        
        logger.info(
            f"[{self.name}] 전처리기 초기화 완료 "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )
    
    def preprocess(self, text: str) -> str:
        """
        텍스트 전처리 (정제, 정규화 등)
        
        Args:
            text: 전처리할 텍스트
            
        Returns:
            전처리된 텍스트
        """
        if not text:
            return ""
        
        # 1. 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 2. 연속된 줄바꿈을 하나로
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 3. 앞뒤 공백 제거
        text = text.strip()
        
        # 4. 특수 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 텍스트 청크 리스트
        """
        if not text:
            return []
        
        chunks = []
        
        # separator로 먼저 분할
        if self.separator:
            parts = text.split(self.separator)
        else:
            parts = [text]
        
        current_chunk = ""
        
        for part in parts:
            # 현재 청크에 추가해도 크기를 초과하지 않는 경우
            if len(current_chunk) + len(part) + len(self.separator) <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + part
                else:
                    current_chunk = part
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 파트가 chunk_size보다 큰 경우 강제 분할
                if len(part) > self.chunk_size:
                    # 큰 파트를 작은 청크로 분할
                    sub_chunks = self._split_large_text(part)
                    chunks.extend(sub_chunks[:-1])  # 마지막 제외하고 추가
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = part
        
        # 남은 청크 추가
        if current_chunk:
            chunks.append(current_chunk)
        
        # 오버랩 적용
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        logger.debug(f"[{self.name}] 텍스트 분할 완료: {len(chunks)}개 청크")
        
        return chunks
    
    def _split_large_text(self, text: str) -> List[str]:
        """
        큰 텍스트를 강제로 분할
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 텍스트 리스트
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 문장 경계에서 자르기 시도
            if end < len(text):
                # 마침표, 느낌표, 물음표, 줄바꿈에서 자르기
                boundary_chars = ['. ', '! ', '? ', '\n']
                best_cut = end
                
                # 뒤로 최대 100자까지 검색
                search_start = max(start, end - 100)
                for char in boundary_chars:
                    pos = text.rfind(char, search_start, end)
                    if pos > best_cut - 100:  # 너무 앞에서 자르지 않도록
                        best_cut = pos + len(char)
                        break
                
                end = best_cut
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        청크 간 오버랩 적용
        
        Args:
            chunks: 원본 청크 리스트
            
        Returns:
            오버랩이 적용된 청크 리스트
        """
        if not chunks or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            # 이전 청크의 뒷부분을 현재 청크 앞에 추가
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # 이전 청크에서 overlap 크기만큼 가져오기
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # 오버랩 텍스트와 현재 청크 결합
            overlapped_chunk = overlap_text + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def split_by_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """
        문장 단위로 텍스트 분할
        
        Args:
            text: 분할할 텍스트
            max_sentences: 청크당 최대 문장 수
            
        Returns:
            문장 단위로 분할된 청크 리스트
        """
        if not text:
            return []
        
        # 문장 분리 (간단한 정규식 사용)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 현재 청크에 추가해도 크기 제한을 넘지 않는 경우
            if (len(current_chunk) < max_sentences and 
                current_length + sentence_length <= self.chunk_size):
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # 새 청크 시작
                current_chunk = [sentence]
                current_length = sentence_length
        
        # 남은 청크 추가
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.debug(f"[{self.name}] 문장 단위 분할 완료: {len(chunks)}개 청크")
        
        return chunks

