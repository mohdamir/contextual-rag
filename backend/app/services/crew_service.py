from crewai import Agent, Task, Crew
from pydantic import BaseModel
from app.core.llms import get_chatopenai_llm

import logging


class CrewAIConfig(BaseModel):
    verbose: bool = True
    max_iter: int = 3
    enable_prompt_enhancer: bool = True


class CrewService:
    def __init__(self, config: CrewAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = get_chatopenai_llm()

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
        if not self.config.enable_prompt_enhancer:
            self.logger.info("Prompt enhancer disabled; returning original prompt.")
            return original_prompt

        try:
            optimizer = self._create_agent(
                role="Dual Prompt Optimizer",
                goal="Rewrite user input for both semantic retrieval and language model processing",
                backstory=(
                    "Expert in semantic search and prompt engineering. "
                    "Knows how to produce both info-dense prompts for retrieval and rich prompts for generation."
                )
            )

            description = f"""
                Rewrite the original user prompt into two optimized versions:

                1. **retrieval_prompt**: Short, keyword-focused version optimized for semantic search in a vector database. It should contain only the essential concepts, using domain-specific terms.
                2. **llm_prompt**: Clear and structured version of the original prompt, optimized for input to a language model. It can include clarification and be more natural/verbose.

                ### Example Input
                **Original Prompt**:
                Explain the difference between Foreground and Background Intellectual Property in a contract.

                ### Example Output (JSON):
                {{
                "retrieval_prompt": "difference between Foreground and Background Intellectual Property in contracts",
                "llm_prompt": "Explain the difference between Foreground and Background Intellectual Property within a contractual context."
                }}

                Return your response **only as a JSON object** in this format.
                Do not include any commentary or explanation.

                Original Prompt: {original_prompt}
                """


            optimization_task = Task(
                description=description,
                agent=optimizer,
                expected_output="JSON object with retrieval_prompt and llm_prompt.",
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
