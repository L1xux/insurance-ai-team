"""
Customer Insight Agent
=========================
Author: Jin
Date: 2026.02.07
Version: 4.6 (Robust & Parallel & Viz Enforced)

Description:
인구통계 프로파일링, 구매 행동 분석, 타겟 세그먼트 식별을 수행하는 Worker Agent입니다.
병렬 실행 및 입력값 방어 로직이 적용되었으며, 시각화 도구 사용을 위한 프롬프트가 강화되었습니다.
"""
from typing import Dict, Any, Optional, List
import json
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from base.agent.agent_base import WorkerAgent
from base.agent.llm_base import LLMBase
from models.agent_model import AgentResult, AgentPlan
from config.logging_config import logger

# --- 데이터 구조 정의 ---
class InsightReport(BaseModel):
    summary: str = Field(description="분석 결과의 핵심 요약 (3문장 이내)")
    detailed_analysis: str = Field(description="상세 분석 내용 및 근거")
    key_metrics: List[str] = Field(description="주요 발견 지표 목록", default_factory=list)

class CustomerInsightAgent(WorkerAgent):
    """고객 인사이트 분석 Worker Agent"""
    
    CONTEXT_LENGTH_LIMIT = 3000

    def __init__(
        self,
        name: str,
        description: str,
        tools: List[BaseTool],
        llm: Optional[LLMBase] = None
    ):
        super().__init__(name, description, tools, llm)
        
        if self.llm:
            self._setup_chains()
        
        logger.info(f"[{self.name}] 초기화 완료")
    
    def _setup_chains(self) -> None:
        """체인 구성"""
        self.chat_model = self.llm.get_model()
        if not self.chat_model:
            logger.warning(f"[{self.name}] 모델 없음")
            return

        # 1. Planning Chain
        structured_plan_llm = self.chat_model.with_structured_output(
            AgentPlan,
            method="function_calling"
        )
        
        # [수정] 시각화 강제 및 도구 선택 가이드 강화
        plan_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 보험 데이터 분석 및 시각화 전문가입니다.

사용 가능한 도구 목록:
{tools}

[도구 선택 가이드 - 키워드 매칭]
1. "차트", "그래프", "시각화", "도표", "이미지" 요청 시:
   -> 반드시 **'visualization_generator'**를 도구 목록에 포함하세요.
   -> 쿼리 예시: "age distribution", "charges by smoker"
   
2. "인구", "통계", "분포", "가입자":
   -> **'h2022_demographic_analysis'**
   
3. "리스크", "위험", "고위험군", "질병":
   -> **'h2022_risk_analysis'**

[작업 지침]
1. 사용자가 차트를 요청했다면, 분석 도구와 시각화 도구를 **함께** 실행하도록 계획하세요.
   - 예: "인구 통계 분석하고 차트 그려줘" -> tools: [h2022_demographic_analysis, visualization_generator]
2. 각 도구 호출 시 'query' 필드에 구체적인 분석 주제를 문자열로 입력하세요.

AgentPlan 형식:
- reasoning: 도구 조합 전략 및 계획
- tools: Tool 실행 계획 리스트 (tool_name, query 필수)
"""),
            ("user", """요청: {request}

[컨텍스트 정보]
{context_data}""")
        ])
        
        self.plan_chain = plan_prompt | structured_plan_llm
        
        # 2. Synthesis Chain
        structured_report_llm = self.chat_model.with_structured_output(
            InsightReport,
            method="function_calling"
        )
        
        synthesize_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 보험 데이터 분석 전문가입니다.

도구 실행 결과와 생성된 시각화 정보를 종합하여 인사이트 보고서를 작성하세요.
핵심 요약(summary)과 상세 분석(detailed_analysis)을 명확히 분리하여 작성해야 합니다.
"""),
            ("user", """질문: {request}

[이전 맥락]
{context_data}

[도구 실행 결과]
{tool_results}""")
        ])
        
        self.synthesis_chain = synthesize_prompt | structured_report_llm
    
    async def process(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """분석 프로세스 실행"""
        try:
            logger.info(f"[{self.name}] 분석 시작: {message[:50]}...")
            
            # 1. Context 데이터 처리
            previous_results = context.get('previous_results', []) if context else []
            optimized_context = self._optimize_context(previous_results)
            
            # 2. Plan 수립 
            tool_schemas = [tool.get_schema().to_dict() for tool in self.tools.values()]
            plan = await self.plan_chain.ainvoke({
                "tools": json.dumps(tool_schemas, indent=2, ensure_ascii=False),
                "request": message,
                "context_data": optimized_context
            })
            
            logger.info(f"[{self.name}] Plan: {plan.reasoning}")
            
            # 3. Tool 실행 (병렬 처리 적용)
            tool_results = []
            if plan.tools:
                tool_results = await self._execute_tools(plan.tools)
            
            # Fail-Fast
            if not tool_results or all(not r.get('success', False) for r in tool_results):
                logger.warning(f"[{self.name}] 도구 실행 실패 또는 결과 없음")
                return AgentResult(
                    success=False,
                    data={"error": "분석에 필요한 데이터를 확보하지 못했습니다."},
                    error="No valid tool results"
                )

            # 4. 결과 통합
            # 도구 결과를 텍스트로 변환하여 LLM에 전달
            formatted_results = self._format_results(tool_results)
            
            report = await self.synthesis_chain.ainvoke({
                "request": message,
                "context_data": optimized_context,
                "tool_results": formatted_results
            })
            
            # 5. 최종 결과 구성
            final_output = self._construct_final_output(report, tool_results)
            
            return AgentResult(
                success=True,
                data=final_output
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] 분석 실패: {str(e)}")
            return AgentResult(success=False, data=None, error=str(e))

    async def _execute_tools(self, tool_plans: List) -> List[Dict[str, Any]]:
        """Tool 병렬 실행 및 입력값 방어 로직 (Robust Execution)"""
        
        async def run_single_tool(plan) -> Dict[str, Any]:
            tool_name = plan.tool_name
            # [방어 로직] 입력값 전처리
            raw_input = plan.query
            
            if raw_input is None:
                tool_input = "{}" # 빈 JSON 문자열
            elif isinstance(raw_input, dict):
                tool_input = json.dumps(raw_input, ensure_ascii=False)
            else:
                tool_input = str(raw_input)

            tool = self.tools.get(tool_name)
            if not tool:
                logger.warning(f"[{self.name}] Tool 없음: {tool_name}")
                return {'tool_name': tool_name, 'success': False, 'error': "Tool not found"}
            
            logger.info(f"[{self.name}] Tool 실행: {tool_name} (Input: {tool_input[:50]}...)")
            
            try:
                # BaseTool 실행
                if hasattr(tool, 'ainvoke'):
                    tool_result_str = await tool.ainvoke(tool_input)
                    
                    # 결과 파싱 시도 (JSON -> Dict)
                    try:
                        if isinstance(tool_result_str, dict):
                            tool_result = tool_result_str
                        else:
                            tool_result = json.loads(tool_result_str)
                    except (json.JSONDecodeError, TypeError):
                        tool_result = {"success": True, "data": str(tool_result_str)}

                    return {
                        'tool_name': tool_name,
                        'success': tool_result.get('success', True),
                        'data': tool_result, # 전체 데이터 반환
                        'error': tool_result.get('error')
                    }
                else:
                    return {'tool_name': tool_name, 'success': False, 'error': "Async unsupported"}
            except Exception as e:
                logger.error(f"[{self.name}] Tool 실행 오류: {str(e)}")
                return {'tool_name': tool_name, 'success': False, 'error': str(e)}

        # 병렬 실행 (Fan-out)
        tasks = [run_single_tool(p) for p in tool_plans]
        results = await asyncio.gather(*tasks)
        
        return results

    def _format_results(self, results: List[Dict]) -> str:
        """도구 실행 결과를 LLM이 읽기 편한 텍스트로 변환"""
        lines = []
        for r in results:
            name = r.get('tool_name')
            data = r.get('data')
            lines.append(f"[{name} Result]\n{data}\n")
        return "\n".join(lines)

    def _optimize_context(self, previous_results: List[Dict[str, Any]]) -> str:
        """이전 결과 최적화 (기존 로직 유지)"""
        if not previous_results:
            return "이전 맥락 없음"

        full_text_list = []
        for res in previous_results:
            full_res = res.get('output') or res.get('full_result') or res.get('answer')
            if isinstance(full_res, dict):
                full_text_list.append(json.dumps(full_res, ensure_ascii=False))
            else:
                full_text_list.append(str(full_res))
        
        full_context = "\n".join(full_text_list)
        
        if len(full_context) > self.CONTEXT_LENGTH_LIMIT:
            logger.info(f"[{self.name}] Context 길이 초과 -> Summary 사용")
            summary_list = []
            for res in previous_results:
                summary_list.append(str(res.get('summary', '')))
            return "\n".join(summary_list)
            
        return full_context
    
    def _construct_final_output(self, report: InsightReport, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최종 결과 데이터 구조화"""
        
        full_result_lines = [report.detailed_analysis, "\n"]
        
        if report.key_metrics:
            full_result_lines.append("### 주요 지표")
            for metric in report.key_metrics:
                full_result_lines.append(f"- {metric}")
        
        return {
            "summary": report.summary,
            "full_result": "\n".join(full_result_lines),
            "evidence_data": tool_results # 원본 데이터 전달
        }