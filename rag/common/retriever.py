"""
기본 Retriever 구현체
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
임베딩 모델과 벡터 저장소를 조합한 기본 검색기 구현체입니다.
쿼리를 임베딩하고 벡터 저장소에서 유사한 문서를 검색합니다.
"""
from typing import List, Dict, Any, Optional

from base.rag.retriever_base import RetrieverBase, SearchResult
from base.rag.embedding_base import EmbeddingBase
from base.rag.vector_store_base import VectorStoreBase
from config.logging_config import logger


class Retriever(RetrieverBase):
    """기본 검색기 구현체"""
    
    def __init__(
        self,
        embedder: EmbeddingBase,
        vector_store: VectorStoreBase,
        top_k: int = 5,
        name: Optional[str] = None
    ):
        """
        검색기 초기화
        
        Args:
            embedder: 임베딩 모델
            vector_store: 벡터 저장소
            top_k: 반환할 결과 개수
            name: 검색기 이름
        """
        super().__init__(top_k=top_k, name=name)
        
        self.embedder = embedder
        self.vector_store = vector_store
        
        # 차원 일치 확인
        if self.embedder.dimension != self.vector_store.dimension:
            logger.warning(
                f"[{self.name}] 임베더 차원({self.embedder.dimension})과 "
                f"벡터 저장소 차원({self.vector_store.dimension})이 일치하지 않습니다."
            )
        
        logger.info(
            f"[{self.name}] 검색기 초기화 완료 "
            f"(embedder={embedder.model_name}, store={vector_store.name}, top_k={top_k})"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        쿼리에 대한 관련 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수 (None이면 기본값 사용)
            filter_dict: 메타데이터 필터링 조건
            
        Returns:
            검색 결과 리스트
        """
        try:
            if not query or not query.strip():
                logger.warning(f"[{self.name}] 빈 쿼리가 입력되었습니다.")
                return []
            
            # top_k 설정
            k = top_k if top_k is not None else self.top_k
            
            # 쿼리 임베딩
            logger.debug(f"[{self.name}] 쿼리 임베딩 중: '{query[:50]}...'")
            query_vector = self.embedder.embed_text(query)
            
            # 벡터 저장소에서 검색
            logger.debug(f"[{self.name}] 벡터 검색 중 (top_k={k})")
            raw_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                filter_dict=filter_dict
            )
            
            # SearchResult 객체로 변환
            search_results = []
            for doc_id, score, metadata in raw_results:
                content = metadata.get('content', metadata.get('text', ''))
                
                result = SearchResult(
                    content=content,
                    score=score,
                    metadata=metadata,
                    doc_id=doc_id
                )
                search_results.append(result)
            
            self.search_count += 1
            logger.info(f"[{self.name}] 검색 완료: {len(search_results)}개 결과 (검색 횟수: {self.search_count})")
            
            return search_results
            
        except Exception as e:
            logger.error(f"[{self.name}] 검색 실패: {str(e)}")
            raise
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[SearchResult]]:
        """
        여러 쿼리에 대한 일괄 검색
        
        Args:
            queries: 검색 쿼리 리스트
            top_k: 반환할 결과 개수 (None이면 기본값 사용)
            
        Returns:
            각 쿼리에 대한 검색 결과 리스트
        """
        try:
            if not queries:
                logger.warning(f"[{self.name}] 빈 쿼리 리스트가 입력되었습니다.")
                return []
            
            # 각 쿼리에 대해 개별 검색
            batch_results = []
            for query in queries:
                results = self.retrieve(query=query, top_k=top_k)
                batch_results.append(results)
            
            logger.info(f"[{self.name}] 배치 검색 완료: {len(queries)}개 쿼리")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"[{self.name}] 배치 검색 실패: {str(e)}")
            raise
    
    def add_documents(
        self,
        contents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        문서를 임베딩하여 벡터 저장소에 추가
        
        Args:
            contents: 문서 내용 리스트
            metadata_list: 각 문서의 메타데이터 리스트
            ids: 각 문서의 고유 ID 리스트
            
        Returns:
            추가된 문서의 ID 리스트
        """
        try:
            if not contents:
                logger.warning(f"[{self.name}] 추가할 문서가 없습니다.")
                return []
            
            # 메타데이터에 content 추가
            if metadata_list is None:
                metadata_list = [{} for _ in contents]
            
            for idx, content in enumerate(contents):
                if 'content' not in metadata_list[idx]:
                    metadata_list[idx]['content'] = content
            
            # 문서 임베딩
            logger.info(f"[{self.name}] {len(contents)}개 문서 임베딩 중...")
            vectors = self.embedder.embed_texts(contents)
            
            # 벡터 저장소에 추가
            logger.info(f"[{self.name}] 벡터 저장소에 추가 중...")
            doc_ids = self.vector_store.add_vectors(
                vectors=vectors,
                metadata=metadata_list,
                ids=ids
            )
            
            logger.info(f"[{self.name}] 문서 추가 완료: {len(doc_ids)}개")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"[{self.name}] 문서 추가 실패: {str(e)}")
            raise

