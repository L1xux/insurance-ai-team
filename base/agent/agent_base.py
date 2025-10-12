"""
Agent 베이스 인터페이스
=========================
Author: Jin
Date: 2025.10.12
Version: 1.0

Description:
Manager Agent와 Worker Agent의 베이스 인터페이스입니다.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from base.agent.llm_base import LLMBase
from models.agent_model import AgentResult
from config.logging_config import logger
from langchain_core.tools import BaseTool


class AgentBase(ABC):
    """Agent 베이스 클래스"""
    
    def __init__(self, name: str, description: str):
        """
        Agent 초기화
        
        Args:
            name: Agent 이름
            description: Agent 설명
        """
        self.name = name
        self.description = description
        logger.info(f"[Agent] {name} 초기화: {description}")
    
    @abstractmethod
    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        메시지 처리
        
        Args:
            message: 입력 메시지
            context: 컨텍스트
            
        Returns:
            실행 결과
        """
        pass


class WorkerAgent(AgentBase):
    """Worker Agent - Tool을 사용하여 작업 수행"""
    
    def __init__(
        self,
        name: str,
        description: str,
        tools: List[BaseTool],
        llm: Optional[LLMBase] = None
    ):
        """
        Worker Agent 초기화
        
        Args:
            name: Agent 이름
            description: Agent 설명
            tools: 사용 가능한 Tool 리스트
            llm: LLM 인스턴스
        """
        super().__init__(name, description)
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm
        logger.info(f"[WorkerAgent] {name} - Tools: {list(self.tools.keys())}")
    
    @abstractmethod
    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """작업 처리"""
        pass


class ManagerAgent(AgentBase):
    """Manager Agent - plan 수립 및 worker 관리"""
    
    def __init__(
        self,
        name: str,
        description: str,
        worker_agents: List[WorkerAgent],
        llm: LLMBase
    ):
        """
        Manager Agent 초기화
        
        Args:
            name: Agent 이름
            description: Agent 설명
            worker_agents: Worker Agent 리스트
            llm: LLM 인스턴스 (필수)
        """
        super().__init__(name, description)
        self.worker_agents = {agent.name: agent for agent in worker_agents}
        self.llm = llm
        logger.info(f"[ManagerAgent] {name} - Workers: {list(self.worker_agents.keys())}")
    
    @abstractmethod
    async def plan(self, user_request: str) -> List[Dict[str, Any]]:
        """
        작업 계획 수립
        
        Args:
            user_request: 사용자 요청
            
        Returns:
            작업 계획 리스트
        """
        pass
    
    @abstractmethod
    async def process(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """요청 처리 (plan → worker 실행 → 결과 통합)"""
        pass
