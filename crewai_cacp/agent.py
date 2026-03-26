from typing import Any, Callable, Dict, List, Optional

try:
    from crewai import Agent, Task
    from crewai.tools import BaseTool
except ImportError:
    raise ImportError(
        "CrewAI is not installed. Please install it with: pip install crewai"
    )

from cacp_sdk import CacpClient, CacpAgent


class CACPAgent(Agent):
    """CrewAI agent that wraps a CACP agent."""

    def __init__(
        self,
        role: str,
        client: CacpClient,
        agent_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        goal: Optional[str] = None,
        backstory: Optional[str] = None,
        verbose: bool = False,
        allow_delegation: bool = False,
        **kwargs,
    ):
        """
        Initialize a CACP agent for CrewAI.

        Args:
            role: The role of the agent (required by CrewAI)
            client: CACP client instance
            agent_id: ID of the CACP agent (if known)
            capabilities: List of capabilities to discover agent (if agent_id not known)
            goal: Optional goal for the agent
            backstory: Optional backstory for the agent
            verbose: Whether to enable verbose output
            allow_delegation: Whether to allow task delegation
            **kwargs: Additional arguments for Agent
        """
        self._client = client
        self._agent_id = agent_id
        self._capabilities = capabilities or []
        self._cached_agent: Optional[CacpAgent] = None

        # Build description from goal and backstory
        description = goal or ""
        if backstory:
            description += f" {backstory}"

        super().__init__(
            role=role,
            goal=goal or f"Use CACP agent to perform {role} tasks",
            backstory=backstory or f"CACP agent with role: {role}",
            verbose=verbose,
            allow_delegation=allow_delegation,
            **kwargs,
        )

    async def _get_cacp_agent(self) -> Optional[CacpAgent]:
        """Get the CACP agent, discovering it if necessary."""
        if self._cached_agent:
            return self._cached_agent

        if self._agent_id:
            self._cached_agent = await self._client.agents.get(self._agent_id)
        elif self._capabilities:
            agents = await self._client.agents.query_by_capability(self._capabilities)
            if agents:
                self._cached_agent = agents[0]

        return self._cached_agent

    async def execute_task(
        self,
        task: Task,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """
        Execute a CrewAI task using the CACP agent.

        Args:
            task: CrewAI task to execute
            context: Optional context for the task
            tools: Optional tools available to the agent

        Returns:
            Task execution result
        """
        import asyncio

        async def _execute() -> str:
            agent = await self._get_cacp_agent()
            if not agent:
                return f"No CACP agent found with ID {self._agent_id} or capabilities {self._capabilities}"

            task_description = task.description
            if context:
                task_description = f"{task_description}\nContext: {context}"

            try:
                response = await self._client.messaging.rpc_call(
                    to_agent=agent.id,
                    method="process",
                    params={"task": task_description, "context": context}
                )
                return str(response.result)
            except Exception as e:
                return f"Error executing task with CACP agent: {str(e)}"

        return asyncio.run(_execute())

    def __repr__(self) -> str:
        return f"CACPAgent(role={self.role}, agent_id={self._agent_id})"


class CACPTool(BaseTool):
    """CrewAI tool that wraps a CACP agent."""

    name: str = "CACP Agent Tool"
    description: str = "Use a CACP agent to perform a specific task"
    client: CacpClient
    agent_id: Optional[str] = None
    capability: Optional[str] = None
    method: str = "process"

    def __init__(
        self,
        name: str,
        description: str,
        client: CacpClient,
        agent_id: Optional[str] = None,
        capability: Optional[str] = None,
        method: str = "process",
        **kwargs,
    ):
        """
        Initialize a CACP tool for CrewAI.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            client: CACP client instance
            agent_id: ID of the CACP agent
            capability: Capability to discover agent (if agent_id not known)
            method: RPC method to call
            **kwargs: Additional arguments for BaseTool
        """
        super().__init__(name=name, description=description, **kwargs)
        self.client = client
        self.agent_id = agent_id
        self.capability = capability
        self.method = method

    def _run(self, query: str) -> str:
        """Run the tool synchronously."""
        import asyncio

        async def _execute() -> str:
            agent = None
            if self.agent_id:
                agent = await self.client.agents.get(self.agent_id)
            elif self.capability:
                agents = await self.client.agents.query_by_capability([self.capability])
                if agents:
                    agent = agents[0]

            if not agent:
                return f"No CACP agent found"

            try:
                response = await self.client.messaging.rpc_call(
                    to_agent=agent.id,
                    method=self.method,
                    params={"input": query}
                )
                return str(response.result)
            except Exception as e:
                return f"Error calling CACP agent: {str(e)}"

        return asyncio.run(_execute())

    async def _arun(self, query: str) -> str:
        """Run the tool asynchronously."""
        agent = None
        if self.agent_id:
            agent = await self.client.agents.get(self.agent_id)
        elif self.capability:
            agents = await self.client.agents.query_by_capability([self.capability])
            if agents:
                agent = agents[0]

        if not agent:
            return f"No CACP agent found"

        try:
            response = await self.client.messaging.rpc_call(
                to_agent=agent.id,
                method=self.method,
                params={"input": query}
            )
            return str(response.result)
        except Exception as e:
            return f"Error calling CACP agent: {str(e)}"