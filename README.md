> [!IMPORTANT]
> **Historical CACP prototype — retired.** CACP's communication capabilities are now native to [Robots Center](https://robotscenter.net/docs/platform/cross-agent) under `/api/v1` and `/socket`. This repository targets the incompatible retired `/v1` and `/ws/v1` contract and was never published as a supported package. Do not use it for new integrations. Clean official Python, TypeScript, and Elixir SDKs live at [RobotsCenter/sdks](https://github.com/RobotsCenter/sdks).

# crewai-cacp

CrewAI integration for CACP (Cross-Agent Communication Protocol).

This package allows you to integrate CACP agents with CrewAI's role-based agent framework.

## Installation

```bash
pip install crewai-cacp
```

## Quick Start

### Using CACPAgent

```python
from crewai import Agent, Task, Crew, Process
from crewai_cacp import CACPAgent, CacpClient

# Initialize CACP client
client = CacpClient("http://localhost:4001")

# Create a CrewAI agent that wraps a CACP agent
researcher = CACPAgent(
    role="Researcher",
    goal="Conduct thorough research on given topics",
    backstory="You are an expert researcher with access to CACP agents",
    client=client,
    agent_id="researcher-agent-123",
    verbose=True,
)

# Define a task
research_task = Task(
    description="Research the latest developments in AI",
    expected_output="A detailed summary of recent AI developments",
    agent=researcher,
)

# Create a crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True,
)

# Execute the task
result = crew.kickoff()
print(result)
```

### Using Multiple CACP Agents

```python
from crewai import Agent, Task, Crew, Process
from crewai_cacp import CACPAgent, CacpClient

# Initialize CACP client
client = CacpClient("http://localhost:4001")

# Create multiple CrewAI agents wrapping CACP agents
researcher = CACPAgent(
    role="Researcher",
    goal="Conduct thorough research",
    backstory="Expert researcher",
    client=client,
    agent_id="researcher-123",
    verbose=True,
)

writer = CACPAgent(
    role="Writer",
    goal="Write compelling content",
    backstory="Expert writer",
    client=client,
    agent_id="writer-123",
    verbose=True,
)

# Define tasks
research_task = Task(
    description="Research AI trends in 2024",
    agent=researcher,
)

writing_task = Task(
    description="Write a blog post about AI trends",
    agent=writer,
    context=[research_task],
)

# Create a crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

# Execute
result = crew.kickoff()
print(result)
```

## Features

- **CACPAgent**: Wrap CACP agents as CrewAI agents
- **Task Support**: Full integration with CrewAI's task system
- **Async Support**: Full async/await compatibility
- **Multi-Agent Teams**: Seamless support for agent teams
- **Error Handling**: Proper exception handling for CACP errors
- **Context Passing**: Support for task context and dependencies

## Requirements

- Python 3.8+
- crewai >= 0.1.0
- cacp-sdk >= 0.1.0

## Documentation

For more detailed documentation, see the [CACP documentation](https://docs.cacp.ai).

## License

MIT# crewai
