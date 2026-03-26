"""
Example: Using CACPAgent with CrewAI

This example demonstrates how to use a CACP agent as a CrewAI agent.
"""

import asyncio
from crewai import Agent, Task, Crew, Process
from crewai_cacp import CACPAgent, CacpClient


async def main():
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
    result = await asyncio.to_thread(crew.kickoff)
    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(main())