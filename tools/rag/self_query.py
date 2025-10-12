"""
Self-Query RAG Tool
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
자연어 쿼리를 분석하여 검색 쿼리와 메타데이터 필터를 자동으로 추출하는 RAG Tool입니다.
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


class SelfQueryRAGTool(BaseTool):
    """Self-Query RAG Tool"""
    
    name: str = "self_query_rag"
    description: str = """자연어 쿼리를 분석하여 검색 쿼리와 메타데이터 필터를 자동 추출하는 RAG 도구입니다.
    입력 형식: {"query": "자연어 쿼리", "top_k": 5}
    """
    
    retriever: Any = None
    chat_model: Any = None
    metadata_fields: Dict[str, str] = {}
    analysis_chain: Any = None
    
    def __init__(
        self,
        retriever: Retriever,
        llm_model: str = "gpt-4o-mini",
        metadata_fields: Optional[Dict[str, str]] = None
    ):
        """
        Tool 초기화
        
        Args:
            retriever: RAG Retriever
            llm_model: 쿼리 분석용 LLM 모델
            metadata_fields: 사용 가능한 메타데이터 필드 설명
        """
        super().__init__()
        self.retriever = retriever
        self.chat_model = ChatOpenAI(model=llm_model, temperature=0)
        
        # 기본 메타데이터 필드
        self.metadata_fields = metadata_fields or {
            'section': 'Document section (A, B, C, D)',
            'category': 'Data category (Demographics, Insurance, etc.)',
            'page_number': 'Page number (integer)',
            'has_table': 'Contains table (boolean)'
        }
        
        self._setup_chain()
        logger.info(f"[{self.name}] 초기화 완료")
    
    def _setup_chain(self) -> None:
        """쿼리 분석 Chain 구성"""
        
        # 메타데이터 필드 설명
        fields_desc = "\n".join([
            f"- {k}: {v}" for k, v in self.metadata_fields.items()
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 자연어 쿼리를 분석하는 전문가입니다.

사용자의 자연어 쿼리를 분석하여:
1. 실제 검색할 쿼리 (query)
2. 메타데이터 필터 (filters)

를 추출하세요.

사용 가능한 메타데이터 필드:
{fields_desc}

출력 형식 (JSON):
{{{{
    "query": "검색할 실제 쿼리",
    "filters": {{{{
        "field_name": "value"
    }}}}
}}}}

필터가 없으면 filters를 빈 객체로 반환하세요.
"""),
            ("user", "사용자 쿼리: {user_query}")
        ])
        
        self.analysis_chain = prompt | self.chat_model | StrOutputParser()
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Self-Query RAG 실행
        
        Args:
            query: JSON 문자열 또는 자연어 쿼리
            run_manager: callback manager
            
        Returns:
            검색 결과 (JSON 문자열)
        """
        try:
            # JSON 파싱
            try:
                params = json.loads(query)
                user_query = params.get('query', query)
                top_k = params.get('top_k', 5)
            except (json.JSONDecodeError, TypeError):
                user_query = query
                top_k = 5
            
            logger.info(f"[{self.name}] 시작: {user_query[:50]}...")
            
            # 쿼리 분석
            analysis = self._analyze_query(user_query)
            search_query = analysis.get('query', user_query)
            filters = analysis.get('filters', {})
            
            logger.info(f"[{self.name}] 분석 완료 - query: {search_query}, filters: {filters}")
            
            # 검색
            results = self.retriever.retrieve(
                query=search_query,
                top_k=top_k,
                filter_dict=filters if filters else None
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
                    'original_query': user_query,
                    'analyzed_query': search_query,
                    'filters': filters,
                    'results': formatted
                }
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[{self.name}] 실패: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)
    
    def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """쿼리 분석"""
        try:
            result = self.analysis_chain.invoke({"user_query": user_query})
            return json.loads(result)
        except Exception as e:
            logger.warning(f"[{self.name}] 분석 실패, 원본 사용: {str(e)}")
            return {'query': user_query, 'filters': {}}
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="Self-Query RAG - 자연어 쿼리를 분석하여 검색 쿼리와 필터 자동 추출",
            parameters={
                'query': {'type': 'string', 'description': '자연어 쿼리'},
                'top_k': {'type': 'integer', 'description': '반환 문서 개수'}
            },
            required_params=['query']
        )

