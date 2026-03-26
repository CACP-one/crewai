"""
Tests for crewai-cacp package
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from crewai import Task
from crewai_cacp import CACPAgent, CACPTool
from cacp_sdk import CacpClient, CacpAgent


@pytest.fixture
def mock_client():
    """Mock CACP client."""
    client = MagicMock(spec=CacpClient)
    return client


@pytest.fixture
def mock_agent():
    """Mock CACP agent."""
    agent = MagicMock(spec=CacpAgent)
    agent.id = "agent-123"
    agent.name = "Test Agent"
    agent.capabilities = ["research"]
    return agent


@pytest.fixture
def mock_task():
    """Mock CrewAI task."""
    task = MagicMock(spec=Task)
    task.description = "Research AI trends"
    return task


class TestCACPAgent:
    """Tests for CACPAgent class."""

    def test_agent_initialization_with_agent_id(self, mock_client):
        """Test agent initialization with agent ID."""
        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="agent-123",
            goal="Conduct research",
            backstory="Expert researcher"
        )
        assert agent.role == "Researcher"
        assert agent._agent_id == "agent-123"
        assert agent._capabilities == []
        assert agent._cached_agent is None

    def test_agent_initialization_with_capabilities(self, mock_client):
        """Test agent initialization with capabilities."""
        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            capabilities=["research", "analysis"]
        )
        assert agent.role == "Researcher"
        assert agent._agent_id is None
        assert agent._capabilities == ["research", "analysis"]

    @pytest.mark.asyncio
    async def test_get_cacp_agent_by_id(self, mock_client, mock_agent):
        """Test getting CACP agent by ID."""
        mock_client.agents.get = AsyncMock(return_value=mock_agent)

        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="agent-123"
        )

        result = await agent._get_cacp_agent()
        assert result == mock_agent
        assert agent._cached_agent == mock_agent
        mock_client.agents.get.assert_called_once_with("agent-123")

    @pytest.mark.asyncio
    async def test_get_cacp_agent_by_capabilities(self, mock_client, mock_agent):
        """Test getting CACP agent by capabilities."""
        mock_client.agents.query_by_capability = AsyncMock(return_value=[mock_agent])

        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            capabilities=["research"]
        )

        result = await agent._get_cacp_agent()
        assert result == mock_agent
        assert agent._cached_agent == mock_agent
        mock_client.agents.query_by_capability.assert_called_once_with(["research"])

    @pytest.mark.asyncio
    async def test_get_cacp_agent_not_found(self, mock_client):
        """Test getting CACP agent when not found."""
        mock_client.agents.get = AsyncMock(side_effect=Exception("Agent not found"))
        mock_client.agents.query_by_capability = AsyncMock(return_value=[])

        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="nonexistent-id"
        )

        result = await agent._get_cacp_agent()
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_client, mock_agent, mock_task):
        """Test executing a task successfully."""
        mock_client.agents.get = AsyncMock(return_value=mock_agent)
        mock_response = MagicMock()
        mock_response.result = "Research completed: AI is advancing rapidly"
        mock_client.messaging.rpc_call = AsyncMock(return_value=mock_response)

        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="agent-123"
        )

        result = await agent.execute_task(mock_task)
        assert "Research completed" in result

    @pytest.mark.asyncio
    async def test_execute_task_with_context(self, mock_client, mock_agent, mock_task):
        """Test executing a task with context."""
        mock_client.agents.get = AsyncMock(return_value=mock_agent)
        mock_response = MagicMock()
        mock_response.result = "Research with context completed"
        mock_client.messaging.rpc_call = AsyncMock(return_value=mock_response)

        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="agent-123"
        )

        result = await agent.execute_task(mock_task, context="Focus on 2024 trends")
        assert "Research with context completed" in result

    @pytest.mark.asyncio
    async def test_execute_task_no_agent_found(self, mock_client, mock_task):
        """Test executing a task when no agent is found."""
        mock_client.agents.get = AsyncMock(side_effect=Exception("Agent not found"))

        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="nonexistent-id"
        )

        result = await agent.execute_task(mock_task)
        assert "No CACP agent found" in result

    @pytest.mark.asyncio
    async def test_execute_task_error(self, mock_client, mock_agent, mock_task):
        """Test executing a task with error."""
        mock_client.agents.get = AsyncMock(return_value=mock_agent)
        mock_client.messaging.rpc_call = AsyncMock(side_effect=Exception("RPC error"))

        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="agent-123"
        )

        result = await agent.execute_task(mock_task)
        assert "Error executing task" in result

    def test_repr(self, mock_client):
        """Test agent string representation."""
        agent = CACPAgent(
            role="Researcher",
            client=mock_client,
            agent_id="agent-123"
        )
        repr_str = repr(agent)
        assert "CACPAgent" in repr_str
        assert "Researcher" in repr_str
        assert "agent-123" in repr_str


class TestCACPTool:
    """Tests for CACPTool class."""

    def test_tool_initialization_with_agent_id(self, mock_client):
        """Test tool initialization with agent ID."""
        tool = CACPTool(
            name="sentiment_analyzer",
            description="Analyze sentiment",
            client=mock_client,
            agent_id="agent-123"
        )
        assert tool.name == "sentiment_analyzer"
        assert tool.agent_id == "agent-123"
        assert tool.capability is None
        assert tool.method == "process"

    def test_tool_initialization_with_capability(self, mock_client):
        """Test tool initialization with capability."""
        tool = CACPTool(
            name="sentiment_analyzer",
            description="Analyze sentiment",
            client=mock_client,
            capability="sentiment_analysis"
        )
        assert tool.name == "sentiment_analyzer"
        assert tool.agent_id is None
        assert tool.capability == "sentiment_analysis"

    def test_run_sync_success(self, mock_client, mock_agent):
        """Test sync tool execution with success."""
        mock_client.agents.get = AsyncMock(return_value=mock_agent)
        mock_response = MagicMock()
        mock_response.result = "positive"
        mock_client.messaging.rpc_call = AsyncMock(return_value=mock_response)

        tool = CACPTool(
            name="sentiment_analyzer",
            description="Analyze sentiment",
            client=mock_client,
            agent_id="agent-123"
        )

        result = tool._run("I love this product!")
        assert "positive" in result

    def test_run_sync_no_agent_found(self, mock_client):
        """Test sync tool execution when no agent is found."""
        mock_client.agents.get = AsyncMock(side_effect=Exception("Not found"))
        mock_client.agents.query_by_capability = AsyncMock(return_value=[])

        tool = CACPTool(
            name="sentiment_analyzer",
            description="Analyze sentiment",
            client=mock_client,
            capability="nonexistent_capability"
        )

        result = tool._run("test input")
        assert "No CACP agent found" in result

    def test_run_sync_error(self, mock_client, mock_agent):
        """Test sync tool execution with error."""
        mock_client.agents.get = AsyncMock(return_value=mock_agent)
        mock_client.messaging.rpc_call = AsyncMock(side_effect=Exception("RPC error"))

        tool = CACPTool(
            name="sentiment_analyzer",
            description="Analyze sentiment",
            client=mock_client,
            agent_id="agent-123"
        )

        result = tool._run("test input")
        assert "Error calling CACP agent" in result

    @pytest.mark.asyncio
    async def test_arun_success(self, mock_client, mock_agent):
        """Test async tool execution with success."""
        mock_client.agents.get = AsyncMock(return_value=mock_agent)
        mock_response = MagicMock()
        mock_response.result = "positive"
        mock_client.messaging.rpc_call = AsyncMock(return_value=mock_response)

        tool = CACPTool(
            name="sentiment_analyzer",
            description="Analyze sentiment",
            client=mock_client,
            agent_id="agent-123"
        )

        result = await tool._arun("I love this product!")
        assert "positive" in result

    @pytest.mark.asyncio
    async def test_arun_with_capability(self, mock_client, mock_agent):
        """Test async tool execution with capability discovery."""
        mock_client.agents.query_by_capability = AsyncMock(return_value=[mock_agent])
        mock_response = MagicMock()
        mock_response.result = "negative"
        mock_client.messaging.rpc_call = AsyncMock(return_value=mock_response)

        tool = CACPTool(
            name="sentiment_analyzer",
            description="Analyze sentiment",
            client=mock_client,
            capability="sentiment_analysis"
        )

        result = await tool._arun("I hate this product!")
        assert "negative" in result