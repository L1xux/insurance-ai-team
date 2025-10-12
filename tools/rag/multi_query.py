"""
Multi-Query RAG Tool
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
하나의 쿼리를 여러 관점으로 확장하여 검색하는 RAG Tool입니다.
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


class MultiQueryRAGTool(BaseTool):
    """Multi-Query RAG Tool"""
    
    name: str = "multi_query_rag"
    description: str = """하나의 쿼리를 여러 관점으로 확장하여 검색하는 RAG 도구입니다.
    입력 형식: {"query": "검색 쿼리", "top_k": 5}
    """
    
    retriever: Any = None
    num_queries: int = 3
    chat_model: Any = None
    expansion_chain: Any = None
    
    def __init__(
        self,
        retriever: Retriever,
        llm_model: str = "gpt-4o-mini",
        num_queries: int = 3
    ):
        """
        Tool 초기화
        
        Args:
            retriever: RAG Retriever
            llm_model: 쿼리 확장용 LLM 모델
            num_queries: 생성할 쿼리 개수
        """
        super().__init__()
        self.retriever = retriever
        self.num_queries = num_queries
        self.chat_model = ChatOpenAI(model=llm_model, temperature=0)
        self._setup_chain()
        logger.info(f"[{self.name}] 초기화 완료 (queries={num_queries})")
    
    def _setup_chain(self) -> None:
        """쿼리 확장 Chain 구성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 검색 쿼리 확장 전문가입니다.

사용자의 원본 쿼리를 다양한 관점에서 {self.num_queries}개의 다른 쿼리로 변환하세요.
각 쿼리는 서로 다른 측면이나 표현을 사용해야 합니다.

출력 형식 (각 쿼리를 새 줄에):
쿼리1
쿼리2
쿼리3
"""),
            ("user", "원본 쿼리: {query}")
        ])
        
        self.expansion_chain = (
            prompt 
            | self.chat_model 
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.strip().split('\n') if q.strip()])
        )
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Multi-Query RAG 실행
        
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
            
            # 쿼리 확장
            expanded_queries = self._expand_queries(search_query)
            
            # 각 쿼리로 검색
            all_results = []
            for exp_query in expanded_queries:
                results = self.retriever.retrieve(
                    query=exp_query,
                    top_k=top_k,
                    filter_dict=filter_dict
                )
                all_results.extend(results)
            
            # 중복 제거 및 재정렬
            unique_results = self._deduplicate(all_results)
            final_results = sorted(unique_results, key=lambda x: x.score, reverse=True)[:top_k]
            
            formatted = [
                {
                    'content': r.content,
                    'score': r.score,
                    'metadata': r.metadata,
                    'doc_id': r.doc_id
                }
                for r in final_results
            ]
            
            return json.dumps({
                'success': True,
                'data': {
                    'original_query': search_query,
                    'expanded_queries': expanded_queries,
                    'results': formatted
                }
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[{self.name}] 실패: {str(e)}")
            return json.dumps({
                'success': False,
                'error': str(e)
            }, ensure_ascii=False)
    
    def _expand_queries(self, query: str) -> List[str]:
        """쿼리 확장"""
        try:
            expanded = self.expansion_chain.invoke({"query": query})
            return [query] + expanded
        except Exception as e:
            logger.warning(f"[{self.name}] 확장 실패: {str(e)}")
            return [query]
    
    def _deduplicate(self, results: List[Any]) -> List[Any]:
        """중복 제거"""
        seen = set()
        unique = []
        for r in results:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                unique.append(r)
        return unique
    
    def get_schema(self):
        """Tool 스키마 반환"""
        return ToolSchema(
            name=self.name,
            description="Multi-Query RAG - 쿼리를 여러 관점으로 확장하여 검색",
            parameters={
                'query': {'type': 'string', 'description': '검색 쿼리'},
                'top_k': {'type': 'integer', 'description': '반환 문서 개수'},
                'filter_dict': {'type': 'object', 'description': '메타데이터 필터'}
            },
            required_params=['query']
        )

