"""
H2022 PostgreSQL + pgvector Vector Store
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
PostgreSQL의 pgvector 확장을 사용하는 H2022 벡터 저장소입니다.
db_connection.py를 활용하여 벡터 데이터를 저장하고 검색합니다.
H2022Document 모델의 구조화된 필드를 테이블 컬럼으로 저장합니다.
HNSW 인덱스를 사용한 고속 ANN 검색을 지원합니다.
"""
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple

from base.rag.vector_store_base import VectorStoreBase
from database.db_connection import DatabaseManager
from config.logging_config import logger


class H2022VectorStore(VectorStoreBase):
    """PostgreSQL + pgvector 기반 H2022 벡터 저장소"""
    
    def __init__(
        self,
        dimension: int,
        table_name: str = "h2022_vector_embeddings",
        name: Optional[str] = None
    ):
        """
        pgvector 저장소 초기화
        
        Args:
            dimension: 벡터 차원 수
            table_name: 사용할 테이블 이름
            name: 저장소 이름
        """
        super().__init__(dimension=dimension, name=name or "H2022VectorStore")
        
        self.table_name = table_name
        self.db_manager = DatabaseManager()
        
        # 테이블 초기화
        self._initialize_table()
        
        # 벡터 개수 업데이트
        self._update_vector_count()
        
        logger.info(
            f"[{self.name}] PostgreSQL + pgvector 저장소 초기화 완료 "
            f"(table={self.table_name}, dim={self.dimension}, count={self.vector_count})"
        )
    
    def _initialize_table(self) -> None:
        """
        벡터 저장 테이블 초기화
        
        H2022Document 모델의 필드를 반영한 테이블과 인덱스를 생성합니다.
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                # pgvector 확장 활성화
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # 테이블 생성 (H2022Document 모델 반영)
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        embedding vector({self.dimension}),
                        content TEXT,
                        
                        -- H2022Document 핵심 필드
                        page_number INT,
                        total_pages INT,
                        document_type TEXT DEFAULT 'MEPS_HC243_2022',
                        year INT DEFAULT 2022,
                        section TEXT,
                        category TEXT,
                        
                        -- 페이지 특성
                        has_table BOOLEAN DEFAULT FALSE,
                        has_code_values BOOLEAN DEFAULT FALSE,
                        is_variable_definition BOOLEAN DEFAULT FALSE,
                        
                        -- 청크 정보
                        chunk_index INT,
                        chunk_id TEXT,
                        
                        -- 추가 메타데이터 (나머지 정보)
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # HNSW 인덱스 생성 (ANN 검색용)
                # m=16: 각 노드의 최대 연결 수
                # ef_construction=64: 인덱스 구축 시 탐색 범위
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                
                # 메타데이터 GIN 인덱스 생성
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx
                    ON {self.table_name}
                    USING gin (metadata);
                """)
                
                # H2022 특화 인덱스 생성 (자주 사용되는 필터)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_page_idx
                    ON {self.table_name}(page_number);
                """)
                
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_category_idx
                    ON {self.table_name}(category);
                """)
                
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_section_idx
                    ON {self.table_name}(section);
                """)
                
                logger.info(f"[{self.name}] H2022 특화 테이블 및 인덱스 초기화 완료: {self.table_name}")
                
        except Exception as e:
            logger.error(f"[{self.name}] 테이블 초기화 실패: {str(e)}")
            raise
    
    def _update_vector_count(self) -> None:
        """
        저장된 벡터 개수 업데이트
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name};")
                result = cursor.fetchone()
                self.vector_count = result['count'] if result else 0
                
        except Exception as e:
            logger.warning(f"[{self.name}] 벡터 개수 조회 실패: {str(e)}")
            self.vector_count = 0
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        벡터를 저장소에 추가 (H2022Document 필드 자동 추출)
        
        Args:
            vectors: 추가할 벡터 리스트
            metadata: 각 벡터의 메타데이터 리스트 (H2022Document 필드 포함)
            ids: 각 벡터의 고유 ID 리스트
            
        Returns:
            추가된 벡터의 ID 리스트
        """
        try:
            if not vectors:
                logger.warning(f"[{self.name}] 추가할 벡터가 없습니다.")
                return []
            
            # ID 생성
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]
            
            # 메타데이터 기본값
            if metadata is None:
                metadata = [{} for _ in vectors]
            
            # 벡터 차원 검증
            for idx, vector in enumerate(vectors):
                if len(vector) != self.dimension:
                    raise ValueError(
                        f"벡터 {idx}의 차원({len(vector)})이 "
                        f"저장소 차원({self.dimension})과 일치하지 않습니다."
                    )
            
            # 데이터베이스에 삽입 (H2022Document 필드 반영)
            with self.db_manager.get_cursor() as cursor:
                for vector_id, vector, meta in zip(ids, vectors, metadata):
                    # H2022Document 필드 추출
                    content = meta.get('content', '')
                    page_number = meta.get('page_number')
                    total_pages = meta.get('total_pages')
                    document_type = meta.get('document_type', 'MEPS_HC243_2022')
                    year = meta.get('year', 2022)
                    section = meta.get('section')
                    category = meta.get('category')
                    has_table = meta.get('has_table', False)
                    has_code_values = meta.get('has_code_values', False)
                    is_variable_definition = meta.get('is_variable_definition', False)
                    chunk_index = meta.get('chunk_index')
                    chunk_id = meta.get('chunk_id')
                    
                    # 나머지는 metadata JSONB에 저장
                    remaining_meta = {k: v for k, v in meta.items() 
                                     if k not in ['content', 'page_number', 'total_pages', 
                                                  'document_type', 'year', 'section', 'category',
                                                  'has_table', 'has_code_values', 'is_variable_definition',
                                                  'chunk_index', 'chunk_id']}
                    
                    cursor.execute(f"""
                        INSERT INTO {self.table_name} (
                            id, embedding, content,
                            page_number, total_pages, document_type, year, section, category,
                            has_table, has_code_values, is_variable_definition,
                            chunk_index, chunk_id, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            content = EXCLUDED.content,
                            page_number = EXCLUDED.page_number,
                            total_pages = EXCLUDED.total_pages,
                            document_type = EXCLUDED.document_type,
                            year = EXCLUDED.year,
                            section = EXCLUDED.section,
                            category = EXCLUDED.category,
                            has_table = EXCLUDED.has_table,
                            has_code_values = EXCLUDED.has_code_values,
                            is_variable_definition = EXCLUDED.is_variable_definition,
                            chunk_index = EXCLUDED.chunk_index,
                            chunk_id = EXCLUDED.chunk_id,
                            metadata = EXCLUDED.metadata;
                    """, (vector_id, vector, content, 
                          page_number, total_pages, document_type, year, section, category,
                          has_table, has_code_values, is_variable_definition,
                          chunk_index, chunk_id, json.dumps(remaining_meta)))
            
            self.vector_count += len(vectors)
            logger.info(f"[{self.name}] 벡터 추가 완료: {len(vectors)}개 (총 {self.vector_count}개)")
            
            return ids
            
        except Exception as e:
            logger.error(f"[{self.name}] 벡터 추가 실패: {str(e)}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        쿼리 벡터와 유사한 벡터 검색 (H2022 필드 필터링 지원)
        
        Args:
            query_vector: 검색할 쿼리 벡터
            top_k: 반환할 결과 개수
            filter_dict: 메타데이터 필터링 조건 (page_number, category, section 등)
            
        Returns:
            (ID, 유사도 점수, 메타데이터) 튜플 리스트
        """
        try:
            # 벡터 차원 검증
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"쿼리 벡터 차원({len(query_vector)})이 "
                    f"저장소 차원({self.dimension})과 일치하지 않습니다."
                )
            
            # 필터 조건 생성 (H2022 필드 지원)
            filter_clause = ""
            filter_params = []
            if filter_dict:
                conditions = []
                h2022_fields = ['page_number', 'section', 'category', 'year', 
                               'has_table', 'has_code_values', 'is_variable_definition']
                
                for key, value in filter_dict.items():
                    if key in h2022_fields:
                        # H2022 컬럼 필터링 (빠름!)
                        if isinstance(value, bool):
                            conditions.append(f"{key} = %s")
                        elif isinstance(value, (int, float)):
                            conditions.append(f"{key} = %s")
                        else:
                            conditions.append(f"{key} = %s")
                        filter_params.append(value)
                    else:
                        # JSONB 메타데이터 필터링
                        if isinstance(value, str):
                            conditions.append(f"metadata->>%s = %s")
                            filter_params.extend([key, value])
                        elif isinstance(value, (int, float)):
                            conditions.append(f"(metadata->>%s)::numeric = %s")
                            filter_params.extend([key, value])
                        elif isinstance(value, bool):
                            conditions.append(f"(metadata->>%s)::boolean = %s")
                            filter_params.extend([key, value])
                
                filter_clause = "WHERE " + " AND ".join(conditions)
            
            # 유사도 검색 (코사인 유사도, HNSW 인덱스 자동 사용)
            with self.db_manager.get_cursor() as cursor:
                query = f"""
                    SELECT 
                        id,
                        1 - (embedding <=> %s::vector) as similarity,
                        content,
                        page_number,
                        total_pages,
                        document_type,
                        year,
                        section,
                        category,
                        has_table,
                        has_code_values,
                        is_variable_definition,
                        chunk_index,
                        chunk_id,
                        metadata
                    FROM {self.table_name}
                    {filter_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                
                # 파라미터 준비
                search_params = filter_params + [query_vector, query_vector, top_k] if filter_params else [query_vector, query_vector, top_k]
                cursor.execute(query, search_params)
                results = cursor.fetchall()
            
            # 결과 변환 (H2022Document 필드 복원)
            search_results = []
            for row in results:
                # H2022Document 필드를 메타데이터에 포함
                full_metadata = row['metadata'] if isinstance(row['metadata'], dict) else {}
                full_metadata.update({
                    'content': row.get('content', ''),
                    'page_number': row.get('page_number'),
                    'total_pages': row.get('total_pages'),
                    'document_type': row.get('document_type'),
                    'year': row.get('year'),
                    'section': row.get('section'),
                    'category': row.get('category'),
                    'has_table': row.get('has_table'),
                    'has_code_values': row.get('has_code_values'),
                    'is_variable_definition': row.get('is_variable_definition'),
                    'chunk_index': row.get('chunk_index'),
                    'chunk_id': row.get('chunk_id')
                })
                
                search_results.append((
                    row['id'],
                    float(row['similarity']),
                    full_metadata
                ))
            
            logger.info(f"[{self.name}] 검색 완료: {len(search_results)}개 결과 (H2022 필드 활용)")
            
            return search_results
            
        except Exception as e:
            logger.error(f"[{self.name}] 검색 실패: {str(e)}")
            raise
    
    def delete(self, ids: List[str]) -> bool:
        """
        벡터 삭제
        
        Args:
            ids: 삭제할 벡터 ID 리스트
            
        Returns:
            삭제 성공 여부
        """
        try:
            if not ids:
                logger.warning(f"[{self.name}] 삭제할 ID가 없습니다.")
                return True
            
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE id = ANY(%s);
                """, (ids,))
                
                deleted_count = cursor.rowcount
            
            self.vector_count = max(0, self.vector_count - deleted_count)
            logger.info(f"[{self.name}] 벡터 삭제 완료: {deleted_count}개 (총 {self.vector_count}개)")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] 벡터 삭제 실패: {str(e)}")
            return False
    
    def update(
        self,
        ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        벡터 또는 메타데이터 업데이트
        
        Args:
            ids: 업데이트할 벡터 ID 리스트
            vectors: 새로운 벡터 리스트
            metadata: 새로운 메타데이터 리스트
            
        Returns:
            업데이트 성공 여부
        """
        try:
            if not ids:
                logger.warning(f"[{self.name}] 업데이트할 ID가 없습니다.")
                return True
            
            with self.db_manager.get_cursor() as cursor:
                for idx, vector_id in enumerate(ids):
                    updates = []
                    params = []
                    
                    if vectors and idx < len(vectors):
                        updates.append("embedding = %s")
                        params.append(vectors[idx])
                    
                    if metadata and idx < len(metadata):
                        meta = metadata[idx]
                        
                        # H2022 필드 업데이트
                        if 'page_number' in meta:
                            updates.append("page_number = %s")
                            params.append(meta['page_number'])
                        if 'category' in meta:
                            updates.append("category = %s")
                            params.append(meta['category'])
                        if 'section' in meta:
                            updates.append("section = %s")
                            params.append(meta['section'])
                        if 'content' in meta:
                            updates.append("content = %s")
                            params.append(meta['content'])
                        
                        updates.append("metadata = %s")
                        params.append(json.dumps(meta))
                    
                    if updates:
                        params.append(vector_id)
                        query = f"""
                            UPDATE {self.table_name}
                            SET {', '.join(updates)}
                            WHERE id = %s;
                        """
                        cursor.execute(query, params)
            
            logger.info(f"[{self.name}] 벡터 업데이트 완료: {len(ids)}개")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] 벡터 업데이트 실패: {str(e)}")
            return False
    
    def save(self, filepath: str) -> bool:
        """
        저장소를 파일로 저장 (PostgreSQL은 자동 영속화)
        
        Args:
            filepath: 저장할 파일 경로 (사용 안 함)
            
        Returns:
            저장 성공 여부
        """
        logger.info(f"[{self.name}] PostgreSQL + pgvector는 데이터베이스에 자동 저장됩니다.")
        return True
    
    def load(self, filepath: str) -> bool:
        """
        파일에서 저장소 로드 (PostgreSQL은 자동 로드)
        
        Args:
            filepath: 로드할 파일 경로 (사용 안 함)
            
        Returns:
            로드 성공 여부
        """
        logger.info(f"[{self.name}] PostgreSQL + pgvector는 데이터베이스에서 자동 로드됩니다.")
        self._update_vector_count()
        logger.info(f"[{self.name}] 현재 저장된 벡터 수: {self.vector_count}개")
        return True
    
    def clear_table(self) -> bool:
        """
        테이블의 모든 데이터 삭제
        
        Returns:
            삭제 성공 여부
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {self.table_name};")
            
            self.vector_count = 0
            logger.info(f"[{self.name}] 테이블 초기화 완료: {self.table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] 테이블 초기화 실패: {str(e)}")
            return False
    
    def get_vector_by_id(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        ID로 벡터 조회
        
        Args:
            vector_id: 조회할 벡터 ID
            
        Returns:
            (벡터, 메타데이터) 튜플 또는 None
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"""
                    SELECT 
                        embedding, content, page_number, total_pages,
                        document_type, year, section, category,
                        has_table, has_code_values, is_variable_definition,
                        chunk_index, chunk_id, metadata
                    FROM {self.table_name}
                    WHERE id = %s;
                """, (vector_id,))
                
                result = cursor.fetchone()
                
                if result:
                    vector = result['embedding']
                    full_metadata = result['metadata'] if isinstance(result['metadata'], dict) else {}
                    full_metadata.update({
                        'content': result.get('content', ''),
                        'page_number': result.get('page_number'),
                        'total_pages': result.get('total_pages'),
                        'document_type': result.get('document_type'),
                        'year': result.get('year'),
                        'section': result.get('section'),
                        'category': result.get('category'),
                        'has_table': result.get('has_table'),
                        'has_code_values': result.get('has_code_values'),
                        'is_variable_definition': result.get('is_variable_definition'),
                        'chunk_index': result.get('chunk_index'),
                        'chunk_id': result.get('chunk_id')
                    })
                    
                    return (vector, full_metadata)
                
                return None
                
        except Exception as e:
            logger.error(f"[{self.name}] 벡터 조회 실패: {str(e)}")
            return None
    
    def search_by_page(
        self,
        page_number: int,
        top_k: int = 100
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        특정 페이지의 모든 청크 조회
        
        Args:
            page_number: 페이지 번호
            top_k: 반환할 최대 결과 개수
            
        Returns:
            (ID, 메타데이터) 튜플 리스트
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"""
                    SELECT 
                        id, content, page_number, total_pages,
                        document_type, year, section, category,
                        has_table, has_code_values, is_variable_definition,
                        chunk_index, chunk_id, metadata
                    FROM {self.table_name}
                    WHERE page_number = %s
                    ORDER BY chunk_index
                    LIMIT %s;
                """, (page_number, top_k))
                
                results = cursor.fetchall()
            
            # 결과 변환
            search_results = []
            for row in results:
                full_metadata = row['metadata'] if isinstance(row['metadata'], dict) else {}
                full_metadata.update({
                    'content': row.get('content', ''),
                    'page_number': row.get('page_number'),
                    'total_pages': row.get('total_pages'),
                    'document_type': row.get('document_type'),
                    'year': row.get('year'),
                    'section': row.get('section'),
                    'category': row.get('category'),
                    'has_table': row.get('has_table'),
                    'has_code_values': row.get('has_code_values'),
                    'is_variable_definition': row.get('is_variable_definition'),
                    'chunk_index': row.get('chunk_index'),
                    'chunk_id': row.get('chunk_id')
                })
                
                search_results.append((row['id'], full_metadata))
            
            logger.info(f"[{self.name}] 페이지 {page_number} 검색 완료: {len(search_results)}개 청크")
            
            return search_results
            
        except Exception as e:
            logger.error(f"[{self.name}] 페이지 검색 실패: {str(e)}")
            return []

