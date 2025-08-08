from crewai import Agent, Task, Crew
from pydantic import BaseModel
from app.core.llms import get_litellm_ollama

import logging


class CrewAIConfig(BaseModel):
    verbose: bool = True
    max_iter: int = 3
    enable_prompt_enhancer: bool = True


class CrewService:
    def __init__(self, config: CrewAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = get_litellm_ollama()

    def _create_agent(self, role: str, goal: str, backstory: str, tools: list = None) -> Agent:
        """Helper to create agents with Ollama configuration"""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=self.llm,
            verbose=self.config.verbose,
            tools=tools or [],
            allow_delegation=True
        )

    def create_prompt_enhancer_crew(self, original_prompt: str) -> str:
        """Optimize a given prompt through a 2-agent CrewAI pipeline."""
        if not self.config.enable_prompt_enhancer:
            self.logger.info("Prompt enhancer disabled; returning original prompt.")
            return original_prompt

        try:
            optimizer = self._create_agent(
                role="Prompt Optimizer",
                goal="Rewrite prompts for maximum clarity and effectiveness",
                backstory="Specialist in prompt engineering and Wordsmith with deep understanding of LLM capabilities"
            )

            optimization_task = Task(
                description="Rewrite the prompt incorporating all improvements",
                agent=optimizer,
                expected_output="Final enhanced prompt ready for LLM processing",
                llm=self.llm
            )

            crew = Crew(
                agents=[optimizer],
                tasks=[optimization_task],
                verbose=self.config.verbose,
                max_iter=self.config.max_iter,
                llm=self.llm
            )

            result = crew.kickoff()
            return result.strip() if isinstance(result, str) else str(result)

        except Exception as e:
            self.logger.error(f"Prompt enhancement failed: {e}", exc_info=True)
            return original_prompt  # Fallback
