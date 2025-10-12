"""
Customer Insight Agent
=========================
Author: Jin
Date: 2025.10.12
Version: 2.0

Description:
인구통계 프로파일링, 구매 행동 분석, 타겟 세그먼트 식별을 수행하는 Worker Agent입니다.
여러 Tool을 조합하여 사용할 수 있습니다.
"""
from typing import Dict, Any, Optional, List
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from base.agent.agent_base import WorkerAgent
from base.agent.llm_base import LLMBase
from models.agent_model import AgentResult, AgentPlan
from config.logging_config import logger

from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough
                

import json

class CustomerInsightAgent(WorkerAgent):
    """고객 인사이트 분석 Worker Agent"""
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: List[BaseTool],
        llm: Optional[LLMBase] = None
    ):
        """
        Customer Insight Agent 초기화
        
        Args:
            name: Agent 이름
            description: Agent 설명
            tools: 분석 Tool 리스트
            llm: LLM 인스턴스
        """
        super().__init__(name, description, tools, llm) 
        
        # 체인 설정
        if self.llm:
            self._setup_chains()
        
        logger.info(f"[{self.name}] Customer Insight Agent 초기화 완료")
    
    def _setup_chains(self) -> None:
        """체인 구성"""
        
        # LLMBase 구현체로부터 모델 가져오기
        self.chat_model = self.llm.get_model()
        if not self.chat_model:
            logger.warning(f"[{self.name}] 모델 없음 - Chain 구성 불가")
            return

        # Structured output을 위한 Chat Model 설정
        structured_llm = self.chat_model.with_structured_output(
            AgentPlan,
            method="function_calling"  
        )
        
        # Plan Chain
        plan_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 보험 데이터 고객 인사이트 분석 전문가입니다.

사용 가능한 도구 목록:
{tools}

도구 사용 지침:
1. **도구 선택**: 위 도구 목록의 name과 description을 참고하여 적절한 도구를 선택하세요
2. **도구 조합**: 여러 도구를 순차적으로 조합하여 사용할 수 있습니다
   - 예: 데이터 분석 → 시각화 → RAG 인사이트
3. **키워드 매칭**:
   - "이미지", "차트", "그래프", "시각화" → visualization_generator
   - "전문 자료", "문서", "근거" → RAG 도구 (multi_query_rag 등)
   - "인구통계", "연령", "성별" → h2022_demographic_analysis
   - "리스크", "위험", "손해율" → h2022_risk_analysis 또는 h2022_actuarial_analysis
4. **파라미터**: 각 도구의 parameters를 참고하여 적절한 값을 설정하세요

AgentPlan 형식:
- reasoning: 계획 수립 이유 및 도구 조합 전략
- tools: Tool 실행 계획 리스트 (순서대로)
  - tool_name: Tool의 정확한 name (도구 목록 참고)
  - query: 검색 쿼리 (RAG 도구용, Optional)
  - analysis_type: 분석 유형 (분석 도구용, Optional)
  - top_k: 반환 개수 (Optional)
"""),
            ("user", "{request}")
        ])
        
        self.plan_chain = plan_prompt | structured_llm
        logger.info(f"[{self.name}] Plan Chain 구성 완료 (Function Calling)")
    
    async def process(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        고객 인사이트 분석 처리
        
        Args:
            message: 분석 요청
            context: 컨텍스트
            
        Returns:
            분석 결과
        """
        try:
            logger.info(f"[{self.name}] 분석 시작: {message[:50]}...")
            
            if not self.llm:
                return await self._process_without_llm(message, context)
            
            # Context에서 chat history 추출 
            chat_history = context.get('chat_history', []) if context else []
            previous_results = context.get('previous_results', []) if context else []
            
            # 1. Plan 수립 
            plan = await self._create_plan(message)
            
            # 2. Tool 실행
            if plan.tools:
                tool_results = await self._execute_tools(plan.tools)
                

                final_answer = await self._synthesize_results(
                    message, 
                    tool_results, 
                    previous_results,
                    chat_history
                )
                
                return AgentResult(
                    success=True,
                    data={
                        'answer': final_answer,
                        'tool_results': tool_results,
                        'reasoning': plan.reasoning
                    }
                )
            else:
                # Tool 없이 LLM 직접 답변
                logger.info(f"[{self.name}] Tool 없이 LLM 직접 답변")
                answer = await self._direct_answer(message)
                return AgentResult(
                    success=True,
                    data={
                        'answer': answer,
                        'tool_results': [],
                        'reasoning': plan.reasoning
                    }
                )
            
        except Exception as e:
            logger.error(f"[{self.name}] 분석 실패: {str(e)}")
            return AgentResult(success=False, data=None, error=str(e))
    
    async def _process_without_llm(
        self,
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> AgentResult:
        """LLM 없이 처리"""
        parameters = context.get('parameters', {}) if context else {}
        tool_name = parameters.get('tool_name')
        
        if not tool_name:
            return AgentResult(
                success=False,
                data=None,
                error="LLM 없이는 tool_name 필요"
            )
        
        tool = self.tools.get(tool_name)
        if not tool:
            return AgentResult(
                success=False,
                data=None,
                error=f"Tool 없음: {tool_name}"
            )
        
        result = tool.execute(**parameters)
        return AgentResult(
            success=result.success,
            data=result.data,
            error=result.error
        )
    
    async def _create_plan(self, request: str) -> AgentPlan:
        """분석 계획 수립"""
        try:
            tool_schemas = [tool.get_schema().to_dict() for tool in self.tools.values()]
            
            plan: AgentPlan = await self.plan_chain.ainvoke({
                "tools": json.dumps(tool_schemas, indent=2, ensure_ascii=False),
                "request": request
            })
            
            logger.info(f"[{self.name}] Plan: {plan.reasoning}, Tools: {len(plan.tools)}개")
            
            return plan
            
        except Exception as e:
            logger.warning(f"[{self.name}] Plan 수립 실패: {str(e)}")
            return AgentPlan(reasoning='Plan failed', tools=[])
    
    async def _execute_tools(self, tool_plans: List) -> List[Dict[str, Any]]:
        """여러 Tool 순차 실행"""
        results = []
        
        for idx, plan in enumerate(tool_plans, 1):
            tool_name = plan.tool_name
            
            # ToolPlan에서 파라미터 생성 
            tool_input = {}
            if plan.query:
                tool_input['query'] = plan.query
            if plan.analysis_type:
                tool_input['analysis_type'] = plan.analysis_type
            if plan.top_k:
                tool_input['top_k'] = plan.top_k
            
            tool = self.tools.get(tool_name)
            if not tool:
                logger.warning(f"[{self.name}] Tool 없음: {tool_name}")
                results.append({
                    'tool_name': tool_name,
                    'success': False,
                    'error': f"Tool 없음: {tool_name}"
                })
                continue
            
            logger.info(f"[{self.name}] Tool {idx}/{len(tool_plans)} 실행: {tool_name}")
            
            # Tool 실행
            try:
                # BaseTool 체크
                if hasattr(tool, 'invoke'):
                    tool_result_str = await tool.ainvoke(json.dumps(tool_input, ensure_ascii=False))
                    tool_result = json.loads(tool_result_str)
                    results.append({
                        'tool_name': tool_name,
                        'success': tool_result.get('success', True),
                        'data': tool_result.get('data'),
                        'error': tool_result.get('error')
                    })
                else:
                    # Legacy ToolBase
                    tool_result = tool.execute(**tool_input)
                    results.append({
                        'tool_name': tool_name,
                        'success': tool_result.success,
                        'data': tool_result.data,
                        'error': tool_result.error
                    })
            except Exception as e:
                logger.error(f"[{self.name}] Tool 실행 오류: {str(e)}")
                results.append({
                    'tool_name': tool_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _synthesize_results(
        self,
        request: str,
        tool_results: List[Dict[str, Any]],
        previous_results: List[Dict[str, Any]] = None,
        chat_history: List = None
    ) -> str:
        """Tool 실행 결과 통합 및 해석"""
        try:
            # Context 정보 확인
            has_previous = previous_results and len(previous_results) > 0
            has_history = chat_history and len(chat_history) > 0
            
            # Chat history를 텍스트로 변환
            history_text = ""
            if has_history:
                history_lines = []
                for msg in chat_history[-4:]:  # 최근 4개만
                    role = "User" if msg.type == "human" else "Assistant"
                    history_lines.append(f"{role}: {msg.content[:100]}...")
                history_text = "\n".join(history_lines)
            
            if has_previous or has_history:

                synthesize_prompt = ChatPromptTemplate.from_messages([
                    ("system", """당신은 보험 데이터 분석 전문가입니다.

여러 도구의 실행 결과, 이전 작업 결과, 대화 기록을 종합하여 사용자 질문에 답변하세요.
- 대화 맥락을 고려하여 일관된 분석을 제공하세요
- 이전 작업 결과를 참고하여 연속적인 분석을 수행하세요
- 데이터 분석 결과와 RAG 결과를 함께 활용하세요
- 도구 결과에 없는 내용은 추측하지 마세요"""),
                    ("user", """질문: {request}

{context_info}

현재 도구 실행 결과:
{results}

위 모든 정보를 종합하여 질문에 답변하세요.""")
                ])
                
                # Context 구성
                context_parts = []
                if has_history:
                    context_parts.append(f"대화 기록:\n{history_text}")
                if has_previous:
                    context_parts.append(f"이전 작업 결과:\n{json.dumps(previous_results, indent=2, ensure_ascii=False)}")
                context_info = "\n\n".join(context_parts)
                
                # 체인 실행
                chain = (
                    {
                        "request": RunnablePassthrough(),
                        "context_info": lambda _: context_info,
                        "results": lambda _: json.dumps(tool_results, indent=2, ensure_ascii=False)
                    }
                    | synthesize_prompt
                    | self.chat_model
                    | StrOutputParser()
                )
                
                answer = await chain.ainvoke(request)
            else:
                synthesize_prompt = ChatPromptTemplate.from_messages([
                    ("system", """당신은 보험 데이터 분석 전문가입니다.

여러 도구의 실행 결과를 종합하여 사용자 질문에 답변하세요.
- 모든 결과를 통합하여 일관된 답변을 제공하세요
- 도구 결과에 없는 내용은 추측하지 마세요"""),
                    ("user", """질문: {request}

도구 실행 결과:
{results}

위 결과를 종합하여 질문에 답변하세요.""")
                ])
                
                answer = await (synthesize_prompt | self.chat_model | StrOutputParser()).ainvoke({
                    "request": request,
                    "results": json.dumps(tool_results, indent=2, ensure_ascii=False)
                })
            
            logger.info(f"[{self.name}] tool_results: {tool_results}")
            logger.info(f"[{self.name}] 결과 통합 완료: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"[{self.name}] 결과 통합 실패: {str(e)}")
            return "결과를 통합할 수 없습니다."
    
    async def _direct_answer(self, request: str) -> str:
        """Tool 없이 LLM 직접 답변"""
        try:
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 보험 데이터 분석 전문가입니다.
사용자 질문에 전문적으로 답변하세요."""),
                ("user", "{request}")
            ])
            
            answer_chain = answer_prompt | self.chat_model | StrOutputParser()
            answer = await answer_chain.ainvoke({"request": request})
            
            return answer
            
        except Exception as e:
            logger.error(f"[{self.name}] 직접 답변 실패: {str(e)}")
            return "답변을 생성할 수 없습니다."
