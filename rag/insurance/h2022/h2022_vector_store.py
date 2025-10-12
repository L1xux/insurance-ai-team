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
KNN 검색을 지원합니다.
"""
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
        schema_name: str = "meps_2022_vector",
        name: Optional[str] = None
    ):
        """
        pgvector 저장소 초기화
        
        Args:
            dimension: 벡터 차원 수
            table_name: 사용할 테이블 이름
            schema_name: 스키마 이름 (기본값: meps_2022_vector)
            name: 저장소 이름
        """
        super().__init__(dimension=dimension, name=name or "H2022VectorStore")
        
        self.table_name = table_name
        self.schema_name = schema_name
        self.full_table_name = f"{schema_name}.{table_name}"
        self.db_manager = DatabaseManager()
        
        # 스키마 및 테이블 초기화
        self._initialize_schema()
        self._initialize_table()
        
        # 벡터 개수 업데이트
        self._update_vector_count()
        
        logger.info(
            f"[{self.name}] PostgreSQL + pgvector 저장소 초기화 완료 "
            f"(schema={schema_name}, table={table_name}, dim={self.dimension}, count={self.vector_count})"
        )
    
    def _initialize_schema(self) -> None:
        """스키마 생성"""
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
                logger.info(f"[{self.name}] 스키마 초기화 완료: {self.schema_name}")
        except Exception as e:
            logger.error(f"[{self.name}] 스키마 초기화 실패: {str(e)}")
            raise
    
    def _initialize_table(self) -> None:
        """
        벡터 저장 테이블 초기화
        
        H2022Document 모델의 필드를 반영한 테이블과 인덱스를 생성합니다.
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                # pgvector 확장 활성화
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # 테이블 생성 (SERIAL id 사용)
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                        id SERIAL PRIMARY KEY,
                        embedding vector({self.dimension}),
                        content TEXT,
                        
                        page_number INT,
                        total_pages INT,
                        document_type TEXT DEFAULT 'MEPS_HC243_2022',
                        section TEXT,
                        category TEXT,
                        
                        -- 페이지 특성
                        has_table BOOLEAN DEFAULT FALSE,
                        
                        -- 청크 정보
                        chunk_index INT,
                        
                        -- 추가 메타데이터
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # 벡터 인덱스 
                # 정확한 KNN 검색 수행
                logger.info(f"[{self.name}] KNN 검색 모드 사용")
                
                # 메타데이터 GIN 인덱스 생성
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx
                    ON {self.full_table_name}
                    USING gin (metadata);
                """)
                
                # H2022 특화 인덱스 생성
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_page_idx
                    ON {self.full_table_name}(page_number);
                """)
                
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_category_idx
                    ON {self.full_table_name}(category);
                """)
                
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_section_idx
                    ON {self.full_table_name}(section);
                """)
                
                logger.info(f"[{self.name}] H2022 특화 테이블 및 인덱스 초기화 완료: {self.full_table_name}")
                
        except Exception as e:
            logger.error(f"[{self.name}] 테이블 초기화 실패: {str(e)}")
            raise
    
    def _update_vector_count(self) -> None:
        """
        저장된 벡터 개수 업데이트
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) as count FROM {self.full_table_name};")
                result = cursor.fetchone()
                self.vector_count = result['count'] if result else 0
                
        except Exception as e:
            logger.warning(f"[{self.name}] 벡터 개수 조회 실패: {str(e)}")
            self.vector_count = 0
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None  # SERIAL 사용하므로 무시됨
    ) -> List[int]:
        """
        벡터를 저장소에 추가 (H2022Document 필드 자동 추출)
        
        Args:
            vectors: 추가할 벡터 리스트
            metadata: 각 벡터의 메타데이터 리스트 (H2022Document 필드 포함)
            ids: 사용되지 않음 (SERIAL auto-increment 사용)
            
        Returns:
            추가된 벡터의 ID 리스트 (SERIAL로 생성된 integer ID)
        """
        try:
            if not vectors:
                logger.warning(f"[{self.name}] 추가할 벡터가 없습니다.")
                return []
            
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
            
            # 데이터베이스에 배치 삽입 (성능 최적화)
            logger.info(f"[{self.name}] {len(vectors)}개 벡터를 데이터베이스에 배치 삽입 중...")
            
            # 데이터 준비
            insert_data = []
            for vector, meta in zip(vectors, metadata):
                # H2022Document 필드 추출
                content = meta.get('content', '')
                page_number = meta.get('page_number')
                total_pages = meta.get('total_pages')
                document_type = meta.get('document_type', 'MEPS_HC243_2022')
                section = meta.get('section')
                category = meta.get('category')
                has_table = meta.get('has_table', False)
                chunk_index = meta.get('chunk_index')
                
                # 나머지는 metadata JSONB에 저장
                remaining_meta = {k: v for k, v in meta.items() 
                                 if k not in ['content', 'page_number', 'total_pages', 
                                              'document_type', 'section', 'category',
                                              'has_table', 'chunk_index']}
                
                insert_data.append((
                    vector, content,
                    page_number, total_pages, document_type, section, category,
                    has_table, chunk_index, json.dumps(remaining_meta)
                ))
            
            # 배치 INSERT 실행
            inserted_ids = []
            with self.db_manager.get_cursor() as cursor:
                query = f"""
                    INSERT INTO {self.full_table_name} (
                        embedding, content,
                        page_number, total_pages, document_type, section, category,
                        has_table, chunk_index, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """
                
                # 각 행 삽입하고 ID 수집
                for data in insert_data:
                    cursor.execute(query, data)
                    result = cursor.fetchone()
                    if result:
                        inserted_ids.append(result['id'])
                
            logger.info(f"[{self.name}] 배치 삽입 완료")
            
            self.vector_count += len(vectors)
            logger.info(f"[{self.name}] 벡터 추가 완료: {len(vectors)}개 (총 {self.vector_count}개)")
            
            return inserted_ids
            
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
            
            # 필터 조건 생성 
            filter_clause = ""
            filter_params = []
            if filter_dict:
                conditions = []
                h2022_fields = ['page_number', 'section', 'category', 'has_table']
                
                for key, value in filter_dict.items():
                    if key in h2022_fields:
                        # H2022 컬럼 필터링
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
            
            # 유사도 검색 
            with self.db_manager.get_cursor() as cursor:
                query = f"""
                    SELECT 
                        id,
                        1 - (embedding <=> %s::vector) as similarity,
                        content,
                        page_number,
                        total_pages,
                        document_type,
                        section,
                        category,
                        has_table,
                        chunk_index,
                        metadata
                    FROM {self.full_table_name}
                    {filter_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                
                # 파라미터 준비
                search_params = filter_params + [query_vector, query_vector, top_k] if filter_params else [query_vector, query_vector, top_k]
                cursor.execute(query, search_params)
                results = cursor.fetchall()
            
            # 결과 변환
            search_results = []
            for row in results:
                # H2022Document 필드를 메타데이터에 포함
                full_metadata = row['metadata'] if isinstance(row['metadata'], dict) else {}
                full_metadata.update({
                    'content': row.get('content', ''),
                    'page_number': row.get('page_number'),
                    'total_pages': row.get('total_pages'),
                    'document_type': row.get('document_type'),
                    'section': row.get('section'),
                    'category': row.get('category'),
                    'has_table': row.get('has_table'),
                    'chunk_index': row.get('chunk_index')
                })
                
                search_results.append((
                    row['id'],
                    float(row['similarity']),
                    full_metadata
                ))
            
            logger.info(f"[{self.name}] KNN 검색 완료: {len(search_results)}개 결과")
            
            return search_results
            
        except Exception as e:
            logger.error(f"[{self.name}] 검색 실패: {str(e)}")
            raise
    
    def delete(self, ids: List[int]) -> bool:
        """
        벡터 삭제
        
        Args:
            ids: 삭제할 벡터 ID 리스트 (integer)
            
        Returns:
            삭제 성공 여부
        """
        try:
            if not ids:
                logger.warning(f"[{self.name}] 삭제할 ID가 없습니다.")
                return True
            
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"""
                    DELETE FROM {self.full_table_name}
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
        ids: List[int],
        vectors: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        벡터 또는 메타데이터 업데이트
        
        Args:
            ids: 업데이트할 벡터 ID 리스트 (integer)
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
                            UPDATE {self.full_table_name}
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
                cursor.execute(f"TRUNCATE TABLE {self.full_table_name} RESTART IDENTITY;")
            
            self.vector_count = 0
            logger.info(f"[{self.name}] 테이블 초기화 완료: {self.full_table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] 테이블 초기화 실패: {str(e)}")
            return False
    
    def get_vector_by_id(self, vector_id: int) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        ID로 벡터 조회
        
        Args:
            vector_id: 조회할 벡터 ID (integer)
            
        Returns:
            (벡터, 메타데이터) 튜플 또는 None
        """
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute(f"""
                    SELECT 
                        embedding, content, page_number, total_pages,
                        document_type, section, category,
                        has_table, chunk_index, metadata
                    FROM {self.full_table_name}
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
                        'section': result.get('section'),
                        'category': result.get('category'),
                        'has_table': result.get('has_table'),
                        'chunk_index': result.get('chunk_index')
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
    ) -> List[Tuple[int, Dict[str, Any]]]:
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
                        document_type, section, category,
                        has_table, chunk_index, metadata
                    FROM {self.full_table_name}
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
                    'section': row.get('section'),
                    'category': row.get('category'),
                    'has_table': row.get('has_table'),
                    'chunk_index': row.get('chunk_index')
                })
                
                search_results.append((row['id'], full_metadata))
            
            logger.info(f"[{self.name}] 페이지 {page_number} 검색 완료: {len(search_results)}개 청크")
            
            return search_results
            
        except Exception as e:
            logger.error(f"[{self.name}] 페이지 검색 실패: {str(e)}")
            return []

