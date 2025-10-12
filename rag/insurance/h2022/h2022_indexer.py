"""
H2022 인덱서
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
H2022 보험 문서를 인덱싱하는 파이프라인입니다.
문서 로더, 전처리기, 임베딩, 벡터 저장소를 조합하여 전체 인덱싱 프로세스를 수행합니다.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path

from base.rag.indexer_base import IndexerBase
from base.rag.embedding_base import EmbeddingBase
from rag.insurance.h2022.h2022_document_loader import H2022DocumentLoader
from rag.insurance.h2022.h2022_vector_store import H2022VectorStore
from rag.insurance.h2022.h2022_preprocessor import H2022Preprocessor
from config.logging_config import logger


class H2022Indexer(IndexerBase):
    """H2022 인덱서 """
    
    def __init__(
        self,
        embedder: EmbeddingBase,
        vector_store: H2022VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        name: Optional[str] = None
    ):
        """
        H2022 인덱서 초기화
        
        Args:
            embedder: 임베딩 모델
            vector_store: 벡터 저장소
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 겹침 크기
            name: 인덱서 이름
        """
        super().__init__(name=name or "H2022Indexer")
        
        self.embedder = embedder
        self.vector_store = vector_store
        self.document_loader = H2022DocumentLoader()
        self.processor = H2022Preprocessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # 차원 일치 확인
        if self.embedder.dimension != self.vector_store.dimension:
            raise ValueError(
                f"임베더 차원({self.embedder.dimension})과 "
                f"벡터 저장소 차원({self.vector_store.dimension})이 일치하지 않습니다."
            )
        
        logger.info(
            f"[{self.name}] H2022 인덱서 초기화 완료 "
            f"(embedder={embedder.model_name}, chunk_size={chunk_size})"
        )
    
    def index_document(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        단일 문서를 인덱싱
        
        Args:
            filepath: 인덱싱할 문서 파일 경로
            metadata: 문서 메타데이터
            
        Returns:
            인덱싱 결과 (처리된 청크 수, ID 리스트 등)
        """
        try:
            logger.info(f"[{self.name}] 문서 인덱싱 시작: {filepath}")
            
            # 1. 문서 로드
            logger.info(f"[{self.name}] 1단계: 문서 로드")
            documents = self.document_loader.load(filepath)
            
            if not documents:
                logger.warning(f"[{self.name}] 로드된 문서가 없습니다: {filepath}")
                return {
                    'success': False,
                    'message': '로드된 문서가 없습니다',
                    'chunks_count': 0
                }
            
            # 2. 전처리 및 청킹
            logger.info(f"[{self.name}] 2단계: 텍스트 전처리 및 청킹")
            all_chunks = []
            all_metadata = []
            
            for doc in documents:
                # H2022 전용 전처리 및 청킹 (메타데이터 활용)
                chunks = self.processor.create_chunks_with_metadata(
                    text=doc.content,
                    page_metadata=doc.metadata
                )
                
                for chunk in chunks:
                    all_chunks.append(chunk.content)
                    
                    # 메타데이터 병합
                    chunk_meta = chunk.metadata.copy()
                    if metadata:
                        chunk_meta.update(metadata)
                    chunk_meta['content'] = chunk.content
                    
                    all_metadata.append(chunk_meta)
            
            logger.info(f"[{self.name}] 총 {len(all_chunks)}개 청크 생성")
            
            # 3. 임베딩
            logger.info(f"[{self.name}] 3단계: 임베딩 생성")
            vectors = self.embedder.embed_texts(all_chunks)
            
            # 4. 벡터 저장소에 추가
            logger.info(f"[{self.name}] 4단계: 벡터 저장소에 추가")
            doc_ids = self.vector_store.add_vectors(
                vectors=vectors,
                metadata=all_metadata
            )
            
            # 통계 업데이트
            self.indexed_count += 1
            self.total_chunks += len(all_chunks)
            
            result = {
                'success': True,
                'filepath': filepath,
                'documents_count': len(documents),
                'chunks_count': len(all_chunks),
                'doc_ids': doc_ids,
                'vector_store_total': self.vector_store.vector_count
            }
            
            logger.info(
                f"[{self.name}] 문서 인덱싱 완료: {len(documents)}개 문서, "
                f"{len(all_chunks)}개 청크 (총 {self.total_chunks}개 청크)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] 문서 인덱싱 실패: {str(e)}")
            return {
                'success': False,
                'filepath': filepath,
                'error': str(e)
            }
    
    def index_documents(
        self,
        filepaths: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        여러 문서를 일괄 인덱싱
        
        Args:
            filepaths: 인덱싱할 문서 파일 경로 리스트
            metadata_list: 각 문서의 메타데이터 리스트
            
        Returns:
            인덱싱 결과 (총 처리된 청크 수, 성공/실패 개수 등)
        """
        try:
            logger.info(f"[{self.name}] 배치 인덱싱 시작: {len(filepaths)}개 파일")
            
            results = []
            success_count = 0
            total_chunks = 0
            
            for idx, filepath in enumerate(filepaths):
                metadata = None
                if metadata_list and idx < len(metadata_list):
                    metadata = metadata_list[idx]
                
                result = self.index_document(filepath, metadata)
                results.append(result)
                
                if result.get('success'):
                    success_count += 1
                    total_chunks += result.get('chunks_count', 0)
            
            batch_result = {
                'success': True,
                'total_files': len(filepaths),
                'success_count': success_count,
                'failed_count': len(filepaths) - success_count,
                'total_chunks': total_chunks,
                'results': results,
                'vector_store_total': self.vector_store.vector_count
            }
            
            logger.info(
                f"[{self.name}] 배치 인덱싱 완료: "
                f"{success_count}/{len(filepaths)}개 성공, "
                f"{total_chunks}개 청크"
            )
            
            return batch_result
            
        except Exception as e:
            logger.error(f"[{self.name}] 배치 인덱싱 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def rebuild_index(self) -> bool:
        """
        인덱스 재구축 (테이블 초기화)
        
        Returns:
            재구축 성공 여부
        """
        try:
            logger.info(f"[{self.name}] 인덱스 재구축 시작")
            
            # 벡터 저장소 초기화
            success = self.vector_store.clear_table()
            
            if success:
                # 통계 초기화
                self.indexed_count = 0
                self.total_chunks = 0
                logger.info(f"[{self.name}] 인덱스 재구축 완료")
            
            return success
            
        except Exception as e:
            logger.error(f"[{self.name}] 인덱스 재구축 실패: {str(e)}")
            return False

