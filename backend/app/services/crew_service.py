from typing import Dict, Optional
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from app.core.llms import get_llm
from datetime import datetime
import logging

# Configuration Model
class CrewAIConfig(BaseModel):
    verbose: bool = True
    max_iter: int = 3
    enable_prompt_enhancer: bool = True

class OllamaCrewService:
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
    
    def _create_prompt_enhancer_crew(self, original_prompt: str) -> str:
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
    
    def create_research_crew(self, topic: str) -> str:
        if self.config.enable_prompt_enhancer:
            self.logger.info("Enhancing prompt...")
            topic = self._create_prompt_enhancer_crew(topic)
        
        # Create agents
        researcher = self._create_agent(
            role="Senior Researcher",
            goal=f"Research information about {topic}",
            backstory="Expert researcher with attention to detail"
        )
        
        fact_checker = self._create_agent(
            role="Fact Checker",
            goal="Verify all factual claims in research",
            backstory="Meticulous verifier who checks sources"
        )
        
        writer = self._create_agent(
            role="Technical Writer",
            goal=f"Create clear explanations about {topic}",
            backstory="Skilled at simplifying complex topics"
        )
        
        # Create tasks
        research_task = Task(
            description=f"Conduct thorough research on: {topic}",
            agent=researcher,
            expected_output="Detailed research report with sources",
            llm=self.llm
        )
        
        fact_check_task = Task(
            description="Verify all facts in the research",
            agent=fact_checker,
            expected_output="Verified report with corrections",
            context=[research_task],
            llm=self.llm,
            async_execution=True
        )
        
        writing_task = Task(
            description="Create comprehensive explanation for end-users",
            agent=writer,
            expected_output="Well-structured document with clear explanations",
            context=[fact_check_task],
            llm=self.llm
        )

        crew = Crew(
            agents=[researcher, fact_checker, writer],
            tasks=[research_task, fact_check_task, writing_task],
            verbose=self.config.verbose,
            llm=self.llm
        )
        
        return crew.kickoff()
