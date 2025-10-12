"""
HyDE (Hypothetical Document Embeddings) RAG Tool
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
질문에 대한 가상의 답변을 생성하고, 그 답변으로 검색하는 RAG Tool입니다.
"""
from typing import Dict, Any, Optional
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI

from rag.common.retriever import Retriever
from config.logging_config import logger
from models.tool_model import ToolSchema


class HyDERAGTool(BaseTool):
    """HyDE RAG Tool"""
    
    name: str = "hyde_rag"
    description: str = """질문에 대한 가상의 답변을 생성하여 검색하는 RAG 도구입니다.
    입력 형식: {"query": "검색 쿼리", "top_k": 5}
    """
    
    retriever: Any = None
    chat_model: Any = None
    generation_chain: Any = None
    
    def __init__(
        self,
        retriever: Retriever,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Tool 초기화
        
        Args:
            retriever: RAG Retriever
            llm_model: 가상 문서 생성용 LLM 모델
        """
        super().__init__()
        self.retriever = retriever
        self.chat_model = ChatOpenAI(model=llm_model, temperature=0.7)
        self._setup_chain()
        logger.info(f"[{self.name}] 초기화 완료")
    
    def _setup_chain(self) -> None:
        """가상 문서 생성 Chain 구성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 보험 데이터 전문가입니다.

아래 질문에 대한 답변을 작성하세요. 
실제 데이터가 없어도 괜찮습니다. 질문에 맞는 전문적인 답변을 작성하세요.
답변은 상세하고 구체적으로 작성하세요."""),
            ("user", "질문: {query}")
        ])
        
        self.generation_chain = prompt | self.chat_model | StrOutputParser()
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        HyDE RAG 실행
        
        Args:
            query: JSON 문자열 또는 검색 쿼리
            run_manager: callback manager
            
        Returns:
            검색 결과 (JSON 문자열)
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
            
            # 가상 문서 생성
            hypothetical_doc = self._generate_hypothetical_doc(search_query)
            logger.info(f"[{self.name}] 가상 문서 생성 완료")
            
            # 가상 문서로 검색
            results = self.retriever.retrieve(
                query=hypothetical_doc,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            formatted = [
                {
                    'content': r.content,
                    'score': r.score,
                    'metadata': r.metadata,
                    'doc_id': r.doc_id
                }
                for r in results
            ]
            
            return json.dumps({
                'success': True,
                'data': {
                    'original_query': search_query,
                    'hypothetical_document': hypothetical_doc,
                    'results': formatted
                }
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[{self.name}] 실패: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)
    
    def _generate_hypothetical_doc(self, query: str) -> str:
        """가상 문서 생성"""
        try:
            return self.generation_chain.invoke({"query": query})
        except Exception as e:
            logger.warning(f"[{self.name}] 생성 실패, 원본 사용: {str(e)}")
            return query
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="HyDE RAG - 가상의 답변을 생성하여 검색",
            parameters={
                'query': {'type': 'string', 'description': '검색 쿼리'},
                'top_k': {'type': 'integer', 'description': '반환 문서 개수'},
                'filter_dict': {'type': 'object', 'description': '메타데이터 필터'}
            },
            required_params=['query']
        )

