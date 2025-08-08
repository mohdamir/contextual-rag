from crewai import Agent, Task, Crew
from pydantic import BaseModel
from app.core.llms import get_llm
import logging

# Configuration Model
class CrewAIConfig(BaseModel):
    verbose: bool = True
    max_iter: int = 3
    enable_prompt_enhancer: bool = True

class CrewService:
    def __init__(self, config: CrewAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = get_llm
        
    def _create_agent(self, role: str, goal: str, backstory: str, tools: list = []) -> Agent:
        """Helper to create agents with Ollama configuration"""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm,
            verbose=self.config.verbose,
            tools=tools,
            allow_delegation=True
        )
    
    def create_prompt_enhancer_crew(self, original_prompt: str) -> str:
        analyzer = self._create_agent(
            role="Prompt Analyst",
            goal="Identify weaknesses and areas for improvement in prompts",
            backstory="Specialist in prompt engineering and LLM communication"
        )
        
        optimizer = self._create_agent(
            role="Prompt Optimizer",
            goal="Rewrite prompts for maximum clarity and effectiveness",
            backstory="Wordsmith with deep understanding of LLM capabilities"
        )
        
        analysis_task = Task(
            description=f"Analyze this prompt for improvements: '{original_prompt}'",
            agent=analyzer,
            expected_output="List of specific improvement suggestions",
            llm=self.llm
        )
        
        optimization_task = Task(
            description="Rewrite the prompt incorporating all improvements",
            agent=optimizer,
            expected_output="Final enhanced prompt ready for LLM processing",
            context=[analysis_task],
            llm=self.llm
        )
        
        crew = Crew(
            agents=[analyzer, optimizer],
            tasks=[analysis_task, optimization_task],
            verbose=self.config.verbose,
            llm=self.llm
        )
        
        return crew.kickoff()

