"""
Contextual Compression RAG Tool
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
검색된 문서를 쿼리와 관련된 부분만 압축하여 반환하는 RAG Tool입니다.
"""
from typing import Dict, Any, List, Optional
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI

from rag.common.retriever import Retriever
from config.logging_config import logger
from models.tool_model import ToolSchema


class ContextualCompressionRAGTool(BaseTool):
    """Contextual Compression RAG Tool"""
    
    name: str = "contextual_compression_rag"
    description: str = """검색된 문서를 쿼리 관련 부분만 압축하여 반환하는 RAG 도구입니다.
    입력 형식: {"query": "검색 쿼리", "top_k": 5}
    """
    
    retriever: Any = None
    chat_model: Any = None
    compression_chain: Any = None
    
    def __init__(
        self,
        retriever: Retriever,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Tool 초기화
        
        Args:
            retriever: RAG Retriever
            llm_model: 압축용 LLM 모델
        """
        super().__init__()
        self.retriever = retriever
        self.chat_model = ChatOpenAI(model=llm_model, temperature=0)
        self._setup_chain()
        logger.info(f"[{self.name}] 초기화 완료")
    
    def _setup_chain(self) -> None:
        """문서 압축 Chain 구성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 문서 압축 전문가입니다.

주어진 문서에서 질문과 관련된 부분만 추출하세요.
관련 없는 부분은 제거하고, 핵심 정보만 남기세요.
원문의 의미를 유지하면서 간결하게 압축하세요."""),
            ("user", """질문: {query}

문서:
{document}

위 문서에서 질문과 관련된 부분만 추출하세요.""")
        ])
        
        self.compression_chain = prompt | self.chat_model | StrOutputParser()
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Contextual Compression RAG 실행
        
        Args:
            query: JSON 문자열 또는 검색 쿼리
            run_manager:callback manager
            
        Returns:
            압축된 검색 결과 (JSON 문자열)
        """
        try:
            # JSON 파싱
            try:
                params = json.loads(query)
                search_query = params.get('query', query)
                top_k = params.get('top_k', 5)
                filter_dict = params.get('filter_dict')
            except (json.JSONDecodeError, TypeError):
                search_query = query
                top_k = 5
                filter_dict = None
            
            logger.info(f"[{self.name}] 시작: {search_query[:50]}...")
            
            # 문서 검색
            results = self.retriever.retrieve(
                query=search_query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # 각 문서 압축
            compressed_results = []
            for r in results:
                compressed_content = self._compress_document(search_query, r.content)
                compressed_results.append({
                    'original_content': r.content,
                    'compressed_content': compressed_content,
                    'score': r.score,
                    'metadata': r.metadata,
                    'doc_id': r.doc_id
                })
            
            logger.info(f"[{self.name}] 압축 완료: {len(compressed_results)}개")
            
            return json.dumps({
                'success': True,
                'data': {
                    'query': search_query,
                    'results': compressed_results
                }
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[{self.name}] 실패: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)
    
    def _compress_document(self, query: str, document: str) -> str:
        """문서 압축"""
        try:
            return self.compression_chain.invoke({
                "query": query,
                "document": document
            })
        except Exception as e:
            logger.warning(f"[{self.name}] 압축 실패, 원본 사용: {str(e)}")
            return document
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="Contextual Compression RAG - 검색된 문서를 쿼리 관련 부분만 압축",
            parameters={
                'query': {'type': 'string', 'description': '검색 쿼리'},
                'top_k': {'type': 'integer', 'description': '반환 문서 개수'},
                'filter_dict': {'type': 'object', 'description': '메타데이터 필터'}
            },
            required_params=['query']
        )

