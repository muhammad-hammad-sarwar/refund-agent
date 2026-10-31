"""Evaluation LLM module for DeepEval metrics using Groq API.

Provides a DeepEval-compatible LLM wrapper using ChatOpenAI with Groq's API endpoint.
"""

import os

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


class GroqEvalLLM(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper for Groq models via ChatOpenAI."""

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self._model_name = model
        api_key = os.getenv("GROQ_API_KEY_EVAL")
        if not api_key:
            raise ValueError(
                "Neither GROQ_API_KEY_EVAL nor GROQ_API_KEY environment variable is set"
            )

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.0,
            max_tokens=2048,
        )

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"groq/{self._model_name}"


# Backward compatibility aliases
OpenRouterEvalLLM = GroqEvalLLM
EvalLLM = GroqEvalLLM


def get_eval_llm(model: str = "openai/gpt-oss-120b") -> GroqEvalLLM:
    """Create GroqEvalLLM wrapper for DeepEval."""
    return GroqEvalLLM(model=model)
