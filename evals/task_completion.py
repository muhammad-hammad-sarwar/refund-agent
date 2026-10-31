"""Task Completion Evaluation Metric for the refund agent.

This module evaluates whether the agent successfully completes the task described in each Golden.
Task completion is defined as: the agent's final response addresses the user's request appropriately
and reaches a terminal state (continue_conversation=False for completed tasks, or appropriate
clarification/continuation for ongoing tasks).

Uses Groq via LangChain's ChatOpenAI for evaluation, wrapped in a DeepEvalBaseLLM adapter.
"""

import os
import sys

# Add parent directory to path to import agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent.agent import RefundAgent
from evals.dataset import GOLDENS
from evals.eval_llm import get_eval_llm

# Task Completion Criteria:
# A task is considered COMPLETE if:
# 1. For refund requests: Agent creates a ticket (success) OR correctly denies with explanation (policy/expiration/not found)
# 2. For clarification requests: Agent asks for missing information appropriately
# 3. For policy queries: Agent explains policy and asks for user_id to proceed
# 4. For conversation end: Agent responds politely and ends conversation
# 5. The agent's response demonstrates understanding and appropriate action for the scenario

TASK_COMPLETION_CRITERIA = """
Evaluate whether the agent successfully completed the task based on the user's input and expected outcome.

Score on a scale of 0.0 to 1.0 based on how well the agent's response matches the expected outcome.

A task is COMPLETE (score >= 0.8) if the agent's response:
- For successful refund requests: Creates a refund ticket with the CORRECT amount and lists the CORRECT items
- For item not found: Clearly states the item was not found in purchase history (does NOT create a ticket for that item)
- For expired items (outside 30-day window): Clearly denies the refund explaining it's outside the 30-day window (does NOT create a ticket for expired items)
- For mixed valid/expired: Processes valid items, denies expired items, creates ticket only for valid amount
- For invalid user: States user not found, does not create ticket
- For clarification needed (no items specified): Asks which item(s) to refund
- For policy questions: Explains the 30-day policy AND mentions items must be in original condition AND asks for user ID
- For conversation end: Responds politely and indicates end
- For typo in item name: Notes the item was not found and suggests the correct name
- For confirm ticket creation: Creates the ticket (this requires context from prior conversation)

Key evaluation principles:
1. The agent MUST correctly identify eligibility (within 30 days = eligible, outside = not eligible)
2. The agent MUST NOT create tickets for items that are not eligible (expired, not purchased)
3. Partial refunds are correct when some items are eligible and some are not
4. Exact wording doesn't need to match - semantic correctness matters
5. Agent errors (like API errors) should score 0.0

A task is INCOMPLETE (score < 0.5) if:
- The agent creates a ticket for an item that should be denied (expired or not purchased)
- The agent fails to mention the 30-day policy when asked
- The agent fails to ask for clarification when items aren't specified
- The agent gives wrong refund amounts
- The agent hallucinates items or prices
- The agent ignores the user's request entirely

Score as a continuous value 0.0-1.0 based on semantic alignment with expected outcome.
"""


def create_task_completion_metric(threshold: float = 0.8):
    """Create a GEval metric for task completion evaluation."""
    eval_llm = get_eval_llm()

    return GEval(
        name="TaskCompletion",
        criteria=TASK_COMPLETION_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=threshold,
        model=eval_llm,
    )


def run_agent_on_goldens(goldens) -> list[LLMTestCase]:
    """Run the refund agent on all goldens and collect actual outputs."""
    test_cases = []

    for i, golden in enumerate(goldens):
        print(f"\n[{i + 1}/{len(goldens)}] Running agent on: {golden.input[:80]}...")

        # Create fresh agent for each test case
        agent = RefundAgent(thread_id=f"eval_{i}")

        try:
            # Process the input
            answer = agent.process_message(golden.input)
            actual_output = answer.response

            # Convert expected_tools format if needed
            if golden.expected_tools:
                # GEval doesn't use expected_tools directly, but we include for reference
                pass

            test_case = LLMTestCase(
                input=golden.input,
                actual_output=actual_output,
                expected_output=golden.expected_output,
                context=golden.context,
            )
            test_cases.append(test_case)

            print(f"  Actual: {actual_output[:100]}...")

        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            # Create test case with error as actual output - will score 0
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=f"AGENT ERROR: {type(e).__name__}: {e!s}",
                expected_output=golden.expected_output,
                context=golden.context,
            )
            test_cases.append(test_case)

    return test_cases


def run_task_completion_evaluation():
    """Run task completion evaluation on all goldens."""
    eval_model = "openai/gpt-oss-120b"
    print("=" * 70)
    print("TASK COMPLETION EVALUATION")
    print("=" * 70)
    print(f"Model: groq/{eval_model}")
    print("Threshold: 0.8")
    print(f"Total Goldens: {len(GOLDENS)}")
    print("=" * 70)

    # Check API key
    if not (os.getenv("GROQ_API_KEY_EVAL") or os.getenv("GROQ_API_KEY")):
        print("\nERROR: GROQ_API_KEY_EVAL or GROQ_API_KEY not set in environment.")
        print("Add GROQ_API_KEY_EVAL to .env file or export GROQ_API_KEY_EVAL=your_key")
        return None

    # Run agent on all goldens
    print("\n[Phase 1] Running agent on all test cases...")
    test_cases = run_agent_on_goldens(GOLDENS)

    # Create metric
    print("\n[Phase 2] Evaluating task completion...")
    metric = create_task_completion_metric(threshold=0.8)

    # Evaluate each test case
    results = []
    passed = 0
    failed = 0
    errors = 0

    for i, test_case in enumerate(test_cases):
        print(f"\n  Evaluating test case {i + 1}/{len(test_cases)}...")
        try:
            metric.measure(test_case)
            score = metric.score
            reason = metric.reason

            is_success = score >= 0.8
            if is_success:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            # Check if actual output contains error
            if "AGENT ERROR" in test_case.actual_output:
                errors += 1
                status = "ERROR"

            results.append(
                {
                    "index": i + 1,
                    "input": GOLDENS[i].input,
                    "expected": GOLDENS[i].expected_output,
                    "actual": test_case.actual_output,
                    "score": score,
                    "reason": reason,
                    "status": status,
                }
            )

            print(f"    {status} | Score: {score:.2f} | {reason[:100]}...")

        except Exception as e:  # noqa: BLE001
            print(f"    ERROR during evaluation: {type(e).__name__}: {e}")
            failed += 1
            errors += 1
            results.append(
                {
                    "index": i + 1,
                    "input": GOLDENS[i].input,
                    "expected": GOLDENS[i].expected_output,
                    "actual": test_case.actual_output,
                    "score": 0.0,
                    "reason": f"Evaluation error: {type(e).__name__}: {e!s}",
                    "status": "ERROR",
                }
            )

    # Summary
    total = len(results)
    completion_rate = passed / total if total > 0 else 0.0

    print("\n" + "=" * 70)
    print("TASK COMPLETION EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Cases: {total}")
    print(f"Passed:           {passed}")
    print(f"Failed:           {failed}")
    print(f"Errors:           {errors}")
    print(f"Completion Rate:  {completion_rate:.2%}")
    print("Threshold:        0.80")
    print(f"Overall:          {'PASS' if completion_rate >= 0.8 else 'FAIL'}")
    print("=" * 70)

    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 70)
    for r in results:
        print(f"\nTest {r['index']}: {r['status']} (score: {r['score']:.2f})")
        print(f"  Input:    {r['input'][:80]}...")
        print(f"  Expected: {r['expected'][:80]}...")
        print(f"  Actual:   {r['actual'][:80]}...")
        print(f"  Reason:   {r['reason'][:120]}...")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "completion_rate": completion_rate,
        "results": results,
    }


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    run_task_completion_evaluation()
