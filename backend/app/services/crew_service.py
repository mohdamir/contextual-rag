from typing import List, Dict
from crewai import Agent, Task, Crew
from pydantic import BaseModel

class CrewAIConfig(BaseModel):
    model: str = "llama3"
    verbose: bool = True
    max_iter: int = 3

class CrewAIService:
    def __init__(self, config: CrewAIConfig):
        self.config = config
        
    def create_research_crew(self, topic: str) -> str:
        # Define agents
        researcher = Agent(
            role="Senior Researcher",
            goal=f"Research and provide information about {topic}",
            backstory="An expert researcher with years of experience",
            verbose=self.config.verbose,
            allow_delegation=False
        )
        
        writer = Agent(
            role="Technical Writer",
            goal=f"Write clear and concise answers about {topic}",
            backstory="A skilled technical writer who simplifies complex topics",
            verbose=self.config.verbose,
            allow_delegation=False
        )
        
        # Define tasks
        research_task = Task(
            description=f"Research {topic} and gather key information",
            agent=researcher,
            expected_output=f"A detailed research report on {topic}"
        )
        
        writing_task = Task(
            description=f"Write a clear answer about {topic} based on the research",
            agent=writer,
            expected_output=f"A well-written paragraph explaining {topic}"
        )
        
        # Create and run crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            verbose=self.config.verbose,
            max_iter=self.config.max_iter
        )
        
        result = crew.kickoff()
        return result