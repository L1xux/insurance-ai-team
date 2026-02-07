"""
Insurance Manager Agent
=========================
Author: Jin
Date: 2026.02.07
Version: 5.0 (Final Correction)

Description:
보험 분석 및 기획을 총괄하는 Manager Agent입니다.
Ragas 평가는 외부에서 수행할 수 있도록 결과와 출처 데이터를 멤버 변수에 저장합니다.
Worker에게 Summary와 Full Result를 모두 전달하여 Worker가 스스로 컨텍스트를 최적화하도록 지원합니다.
"""
from typing import Dict, Any, Optional, List
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from base.agent.agent_base import ManagerAgent, WorkerAgent
from base.agent.llm_base import LLMBase
from config.logging_config import logger
from models.agent_model import AgentResult, ExecutionPlan

class InsuranceManagerAgent(ManagerAgent):
    """보험 데이터 분석 및 기획 Manager Agent"""
    
    MAX_ITERATIONS = 5

    def __init__(
        self,
        name: str,
        description: str,
        worker_agents: List[WorkerAgent],
        llm: LLMBase
    ):
        """Manager Agent 초기화"""
        super().__init__(name, description, worker_agents, llm) 
        
        # 외부 접근을 위한 결과 저장소
        self.result: Optional[str] = None
        self.sources: List[Dict[str, Any]] = []
        self.evidence: List[str] = []
        
        self._setup_chains()
        logger.info(f"[{self.name}] 초기화 완료")
    
    def _setup_chains(self) -> None:
        """체인 구성"""
        self.chat_model = self.llm.get_model()
        if not self.chat_model:
            logger.warning(f"[{self.name}] 모델 없음")
            return
        
        # Worker Agent 정보
        worker_info = self._get_worker_info_text()
        
        # Structured output 설정
        structured_llm = self.chat_model.with_structured_output(
            ExecutionPlan,
            method="function_calling"
        )
        
        # Planning Prompt
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 유능한 보험 프로젝트 매니저입니다.
당신의 목표는 사용자 요청을 해결하기 위해 하위 전문가들을 적절히 조합하여 업무를 배분하는 것입니다.

사용 가능한 전문가 목록:
{worker_info}

작업 계획 수립 가이드:

1. 요청 분석: 사용자가 원하는 것이 단순 분석인지, 기획인지, 아니면 시장 조사인지 파악하십시오.
2. 최적의 전문가 선택:
   - 데이터 분석이 필요하면 customer_analyst를 호출하세요.
   - 시장 조사나 상품 기획이 필요하면 product_planner를 호출하세요.
   - 두 가지 모두 필요하다면, 논리적인 순서를 결정하세요.
3. 불필요한 단계 생략: 필요한 작업만 계획에 포함하세요.

ExecutionPlan 출력 형식:
- analysis: 사용자의 의도와 작업 흐름에 대한 당신의 판단
- tasks: 실행할 작업 목록
  - agent_name: 호출할 Worker 이름
  - description: Worker에게 지시할 구체적인 작업 내용
"""),
            ("user", "{user_request}")
        ])
        
        self.planning_chain = planning_prompt | structured_llm
        logger.info(f"[{self.name}] Planning Chain 구성 완료")
    
    def _get_worker_info_text(self) -> str:
        """Worker Agent 정보를 텍스트로 변환"""
        info_lines = []
        for worker in self.worker_agents.values():
            info_lines.append(f"- [{worker.name}]: {worker.description}")
        return "\n".join(info_lines)
    
    async def plan(self, user_request: str) -> List:
        """작업 계획 수립"""
        try:
            logger.info(f"[{self.name}] 작업 계획 수립 시작")
            
            execution_plan = await self.planning_chain.ainvoke({
                "worker_info": self._get_worker_info_text(),
                "user_request": user_request
            })
            
            logger.info(f"[{self.name}] 계획 수립 완료: {len(execution_plan.tasks)}개 작업")
            
            return execution_plan.tasks
            
        except Exception as e:
            logger.error(f"[{self.name}] 계획 수립 실패: {str(e)}")
            raise
    
    async def process(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        요청 처리: Plan -> Execute -> Aggregate
        """
        try:
            logger.info(f"[{self.name}] 요청 처리 시작: {message[:50]}...")
            
            # 초기화
            self.result = None
            self.sources = []
            self.evidence = []
            
            # 1. Plan 수립
            tasks = await self.plan(message)
            
            results = []
            accumulated_context = [] # Worker 간 공유 메모리
            
            # 2. Worker 실행
            for idx, task in enumerate(tasks, 1):
                # 사이클 제한 체크
                if idx > self.MAX_ITERATIONS:
                    logger.warning(f"[{self.name}] 최대 반복 횟수 초과로 중단")
                    break

                agent_name = task.agent_name
                worker = self.worker_agents.get(agent_name)
                
                if not worker:
                    logger.warning(f"[{self.name}] Worker 없음: {agent_name}")
                    continue
                
                logger.info(f"[{self.name}] Step {idx}: {agent_name} 실행 중...")
                
                # Context 병합 및 전체 데이터 전달
                task_context = {
                    'task_id': task.task_id,
                    'query': task.query,
                    'previous_results': accumulated_context, 
                    **(context or {})
                }
                
                # Worker 실행
                worker_result = await worker.process(task.description, task_context)
                
                # 결과 저장
                result_entry = {
                    'task_id': task.task_id,
                    'agent_name': agent_name,
                    'result': worker_result
                }
                results.append(result_entry)
                
                # 실패 처리
                if not worker_result.success:
                    logger.error(f"[{self.name}] Worker 실행 실패: {worker_result.error}")
                    break

                # 성공 시 데이터 처리
                if worker_result.success and worker_result.data:
                    data = worker_result.data
                    
                    # 1. Context 누적
                    full_res = data.get('full_result') or data.get('answer', '')
                    summary_res = data.get('summary') or data.get('answer', '')
                    
                    accumulated_context.append({
                        'step': idx,
                        'agent': agent_name,
                        'summary': summary_res,
                        'output': full_res 
                    })
                    
                    # 2. Sources 수집
                    if 'sources' in data:
                        self.sources.extend(data['sources'])
                        
                    # 3. Evidence 수집
                    if 'evidence_data' in data:
                        self.evidence.append(str(data['evidence_data']))
            
            # 3. 최종 결과 통합
            final_output = self._aggregate_results(results)
            self.result = final_output.get('answer')
            
            logger.info(f"[{self.name}] 요청 처리 완료")
            
            return AgentResult(
                success=True,
                data=final_output,
                error=None
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] 요청 처리 실패: {str(e)}")
            return AgentResult(success=False, data=None, error=str(e))
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 통합"""
        
        final_answer = "작업을 완료하지 못했습니다."
        
        if results:
            last_result = results[-1]
            if last_result['result'].success:
                res_data = last_result['result'].data
                if isinstance(res_data, dict):
                    # full_result가 있으면 우선 사용
                    final_answer = res_data.get('full_result') or res_data.get('answer', str(res_data))
                else:
                    final_answer = str(res_data)
            else:
                 final_answer = f"작업 중 오류가 발생했습니다: {last_result['result'].error}"

        # 실행 로그 구성
        execution_log = []
        for r in results:
            status = "성공" if r['result'].success else f"실패"
            execution_log.append(f"[{r['agent_name']}] {status}")

        return {
            'answer': final_answer,
            'log': execution_log,
            'details': results,
            'sources': self.sources
        }