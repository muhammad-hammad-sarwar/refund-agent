"""Test runner for DeepEval evaluations of the refund agent."""

import os
import sys

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToolCallingMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall

from evals.eval_llm import get_eval_llm

# Add parent directory to path to import agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import RefundAgent
from evals.dataset import GOLDENS
from evals.step_efficiency import run_step_efficiency_evaluation


class RefundAgentWrapper(DeepEvalBaseLLM):
    """Wrapper to use the RefundAgent with DeepEval."""

    def __init__(self, thread_id: str = "eval"):
        self.agent = RefundAgent(thread_id=thread_id)

    def load_model(self):
        return self.agent

    def generate(self, prompt: str) -> str:
        result = self.agent.process_message(prompt)
        return result.response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "RefundAgent"


def convert_goldens_to_testcases(goldens) -> list[LLMTestCase]:
    """Convert Golden objects to LLMTestCase for DeepEval evaluation."""
    test_cases = []
    for golden in goldens:
        # Convert expected_tools dict to ToolCall objects
        expected_tools = None
        if golden.expected_tools:
            expected_tools = [
                ToolCall(name=tool["name"], arguments=tool["parameters"])
                for tool in golden.expected_tools
            ]

        test_case = LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            context=golden.context,
            expected_tools=expected_tools,
        )
        test_cases.append(test_case)
    return test_cases


def run_evaluations():
    """Run DeepEval evaluations on the refund agent."""
    print("Setting up evaluation...")

    # Convert goldens to test cases
    test_cases = convert_goldens_to_testcases(GOLDENS)

    # Initialize metrics
    eval_llm = get_eval_llm()
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=eval_llm),
        FaithfulnessMetric(threshold=0.7, model=eval_llm),
        ToolCallingMetric(threshold=0.7),  # Checks if expected tools were called
    ]

    print(f"Running evaluation on {len(test_cases)} test cases...")
    print("=" * 60)

    # Run evaluation
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        verbose_mode=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results


def run_all_evaluations():
    """Run all evaluations including step efficiency."""
    print("=" * 70)
    print("RUNNING ALL EVALUATIONS")
    print("=" * 70)

    # Run standard DeepEval metrics
    print("\n[1/2] Running standard DeepEval metrics...")
    run_evaluations()

    # Run step efficiency evaluation
    print("\n[2/2] Running step efficiency evaluation...")
    run_step_efficiency_evaluation()

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()
    run_all_evaluations()
