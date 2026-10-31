"""Step Efficiency Evaluation Metric for the refund agent.

This module evaluates how efficiently the agent completes tasks by measuring
the number of steps (tool calls, LLM calls, iterations) compared to the expected
optimal number of steps for each scenario.

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

# Step Efficiency Criteria:
# A task is considered EFFICIENT if the agent completes it with minimal necessary steps.
# Steps include: LLM calls, tool calls, and conversation iterations.
# The agent should:
# - Not make redundant tool calls
# - Not loop unnecessarily
# - Use the right tools in the right order
# - Complete the task in the minimum number of iterations

STEP_EFFICIENCY_CRITERIA = """
Evaluate whether the agent completed the task efficiently with minimal necessary steps.

Score on a scale of 0.0 to 1.0 based on step efficiency.

A task is EFFICIENT (score >= 0.8) if the agent:
- Completes the task in the minimum expected number of tool calls
- Does not make redundant or unnecessary tool calls
- Does not exceed the expected number of conversation iterations
- Uses the correct tools in the correct order
- Does not call the same tool multiple times with the same parameters unnecessarily
- Does not loop or repeat steps

Expected tool call patterns per scenario type:
- Successful refund (all items valid): find_user -> get_current_date -> calculate_refund -> create_ticket (4 tool calls)
- Item not found: find_user -> get_current_date -> calculate_refund (3 tool calls, no create_ticket)
- Expired item: find_user -> get_current_date -> calculate_refund (3 tool calls, no create_ticket)
- Mixed valid/expired: find_user -> get_current_date -> calculate_refund -> create_ticket (4 tool calls, partial refund)
- Invalid user: find_user (1 tool call, stops early)
- Clarification needed: find_user (1 tool call, asks for items)
- Policy query: No tools needed (0 tool calls, just informational response)
- Conversation end: No tools needed (0 tool calls, polite goodbye)
- Natural user ID: Same as successful refund pattern
- Confirm ticket: create_ticket (1 tool call, if context already established)

Key evaluation principles:
1. Tool calls should match the expected sequence for the scenario
2. No redundant calls (e.g., calling find_user twice, calling calculate_refund multiple times for same items)
3. No unnecessary loops (max iterations should match expected)
4. The agent should not call create_ticket for ineligible items
5. For multi-turn scenarios (like confirm ticket), the agent should maintain context efficiently

A task is INEFFICIENT (score < 0.5) if:
- The agent makes significantly more tool calls than expected (> 2x expected)
- The agent calls the same tool repeatedly with same parameters
- The agent exceeds max iterations (5) without completing
- The agent calls tools in wrong order (e.g., calculate_refund before find_user)
- The agent calls create_ticket when it shouldn't (for expired/not-found items)
- The agent fails to use available context (e.g., re-verifies user in same conversation)

Score as a continuous value 0.0-1.0 based on step efficiency relative to optimal.
"""

# Expected tool call counts for each golden scenario (by index in GOLDENS)
# Based on actual agent behavior: N tool calls + 1 content + 1 structured = N+2 LLM calls
EXPECTED_TOOL_COUNTS = {
    0: 4,   # SCENARIO_1: successful refund all (find_user, get_current_date, calculate_refund, create_ticket)
    1: 1,   # SCENARIO_2: item not found (only find_user - agent doesn't call calc_refund for missing items)
    2: 2,   # SCENARIO_3A: expired item only (find_user, get_current_date)
    3: 3,   # SCENARIO_3B: mixed valid/expired (find_user, get_current_date, calculate_refund for valid item)
    4: 1,   # SCENARIO_4: invalid user (find_user only)
    5: 2,   # SCENARIO_5: expired policy user_003 (find_user, get_current_date)
    6: 4,   # SCENARIO_6: partial refund user_002 (find_user, get_current_date, calculate_refund, create_ticket)
    7: 1,   # SCENARIO_7: no items specified (clarification - find_user)
    8: 4,   # SCENARIO_8: multiple some found some not (find_user, get_current_date, calculate_refund, create_ticket)
    9: 0,   # SCENARIO_9: policy query (no tools)
    10: 0,  # SCENARIO_10: conversation end (no tools)
    11: 4,  # SCENARIO_11: natural user ID (find_user, get_current_date, calculate_refund, create_ticket)
    12: 0,  # SCENARIO_12: confirm ticket creation (no prior context, agent asks for user_id)
    13: 2,  # SCENARIO_13: all items expired (find_user, get_current_date)
}

# Expected max LLM calls for each scenario (N tool calls + 1 content response + 1 structured output = N+2)
EXPECTED_MAX_ITERATIONS = {
    0: 6,   # SCENARIO_1: 4 tools + 2
    1: 3,   # SCENARIO_2: 1 tool + 2
    2: 4,   # SCENARIO_3A: 2 tools + 2
    3: 5,   # SCENARIO_3B: 3 tools + 2
    4: 3,   # SCENARIO_4: 1 tool + 2
    5: 4,   # SCENARIO_5: 2 tools + 2
    6: 6,   # SCENARIO_6: 4 tools + 2
    7: 3,   # SCENARIO_7: 1 tool + 2
    8: 6,   # SCENARIO_8: 4 tools + 2
    9: 2,   # SCENARIO_9: 0 tools + 2 (content + structured)
    10: 2,  # SCENARIO_10: 0 tools + 2
    11: 6,  # SCENARIO_11: 4 tools + 2
    12: 2,  # SCENARIO_12: 0 tools + 2 (asks for info)
    13: 4,  # SCENARIO_13: 2 tools + 2
}


def create_step_efficiency_metric(threshold: float = 0.8):
    """Create a GEval metric for step efficiency evaluation."""
    eval_llm = get_eval_llm()

    return GEval(
        name="StepEfficiency",
        criteria=STEP_EFFICIENCY_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=threshold,
        model=eval_llm,
    )


def run_agent_on_goldens_with_observability(goldens) -> list[tuple[LLMTestCase, dict]]:
    """Run the refund agent on all goldens and collect actual outputs with observability data."""
    test_cases_with_obs = []

    for i, golden in enumerate(goldens):
        print(f"\n[{i + 1}/{len(goldens)}] Running agent on: {golden.input[:80]}...")

        # Create fresh agent for each test case
        agent = RefundAgent(thread_id=f"eval_step_{i}")

        try:
            # Process the input
            answer = agent.process_message(golden.input)
            actual_output = answer.response

            # Get observability summary
            summary = agent.observability.get_summary()

            test_case = LLMTestCase(
                input=golden.input,
                actual_output=actual_output,
                expected_output=golden.expected_output,
                context=golden.context,
            )
            test_cases_with_obs.append((test_case, summary))

            print(f"  Actual: {actual_output[:100]}...")
            print(f"  Tool calls: {summary.get('tool_calls', 0)}, LLM calls: {summary.get('llm_calls', 0)}")

        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            # Create test case with error as actual output - will score 0
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=f"AGENT ERROR: {type(e).__name__}: {e!s}",
                expected_output=golden.expected_output,
                context=golden.context,
            )
            # Empty summary for errors
            summary = {
                "tool_calls": 0,
                "llm_calls": 0,
                "total_duration_ms": 0,
                "tools_used": [],
                "errors": [str(e)],
            }
            test_cases_with_obs.append((test_case, summary))

    return test_cases_with_obs


def calculate_efficiency_score(test_case: LLMTestCase, summary: dict, scenario_index: int) -> tuple[float, str]:
    """Calculate step efficiency score based on observability data and expected values."""
    expected_tool_calls = EXPECTED_TOOL_COUNTS.get(scenario_index, 3)
    expected_max_iterations = EXPECTED_MAX_ITERATIONS.get(scenario_index, 3)

    actual_tool_calls = summary.get("tool_calls", 0)
    actual_llm_calls = summary.get("llm_calls", 0)
    tools_used = summary.get("tools_used", [])
    errors = summary.get("errors", [])

    # If there are agent errors, score 0
    if errors or "AGENT ERROR" in test_case.actual_output:
        return 0.0, "Agent error occurred during execution"

    # Calculate tool call efficiency
    if expected_tool_calls == 0:
        # For scenarios expecting 0 tool calls
        if actual_tool_calls == 0:
            tool_efficiency = 1.0
            tool_reason = "No tool calls made as expected"
        else:
            tool_efficiency = max(0.0, 1.0 - (actual_tool_calls * 0.2))
            tool_reason = f"Expected 0 tool calls, made {actual_tool_calls}"
    else:
        # Calculate ratio of actual to expected
        ratio = actual_tool_calls / expected_tool_calls
        if ratio <= 1.0:
            tool_efficiency = 1.0
            tool_reason = f"Optimal tool calls: {actual_tool_calls}/{expected_tool_calls}"
        elif ratio <= 1.5:
            tool_efficiency = 0.8
            tool_reason = f"Slightly over expected tool calls: {actual_tool_calls}/{expected_tool_calls}"
        elif ratio <= 2.0:
            tool_efficiency = 0.5
            tool_reason = f"Moderately over expected tool calls: {actual_tool_calls}/{expected_tool_calls}"
        else:
            tool_efficiency = 0.2
            tool_reason = f"Significantly over expected tool calls: {actual_tool_calls}/{expected_tool_calls}"

    # Check for redundant tool calls (same tool called multiple times)
    from collections import Counter
    tool_counts = Counter(tools_used)
    redundant_calls = sum(count - 1 for count in tool_counts.values() if count > 1)
    if redundant_calls > 0:
        tool_efficiency *= max(0.5, 1.0 - (redundant_calls * 0.15))
        tool_reason += f" (redundant calls: {redundant_calls})"

    # Check iteration efficiency (llm_calls roughly correlates with iterations)
    if expected_max_iterations > 0:
        iteration_ratio = actual_llm_calls / expected_max_iterations
        if iteration_ratio <= 1.0:
            iteration_efficiency = 1.0
            iter_reason = f"Optimal iterations: {actual_llm_calls}/{expected_max_iterations}"
        elif iteration_ratio <= 1.5:
            iteration_efficiency = 0.8
            iter_reason = f"Slightly over expected iterations: {actual_llm_calls}/{expected_max_iterations}"
        else:
            iteration_efficiency = 0.5
            iter_reason = f"Over expected iterations: {actual_llm_calls}/{expected_max_iterations}"
    else:
        iteration_efficiency = 1.0
        iter_reason = "No iterations expected"

    # Combined efficiency score (weighted)
    efficiency_score = (tool_efficiency * 0.7) + (iteration_efficiency * 0.3)

    reason = f"{tool_reason}; {iter_reason}"

    return efficiency_score, reason


def run_step_efficiency_evaluation():
    """Run step efficiency evaluation on all goldens."""
    eval_model = "openai/gpt-oss-120b"
    print("=" * 70)
    print("STEP EFFICIENCY EVALUATION")
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

    # Run agent on all goldens with observability
    print("\n[Phase 1] Running agent on all test cases with observability...")
    test_cases_with_obs = run_agent_on_goldens_with_observability(GOLDENS)

    # Create GEval metric for qualitative assessment
    print("\n[Phase 2] Evaluating step efficiency with LLM...")
    metric = create_step_efficiency_metric(threshold=0.8)

    # Evaluate each test case
    results = []
    passed = 0
    failed = 0
    errors = 0

    for i, (test_case, summary) in enumerate(test_cases_with_obs):
        print(f"\n  Evaluating test case {i + 1}/{len(test_cases_with_obs)}...")

        # Calculate programmatic efficiency score
        prog_score, prog_reason = calculate_efficiency_score(test_case, summary, i)

        try:
            # Also run LLM-based evaluation for qualitative assessment
            metric.measure(test_case)
            llm_score = metric.score
            llm_reason = metric.reason

            # Combine scores (70% programmatic, 30% LLM)
            combined_score = (prog_score * 0.7) + (llm_score * 0.3)
            combined_reason = f"Programmatic: {prog_reason}. LLM: {llm_reason}"

            is_success = combined_score >= 0.8
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
                    "score": combined_score,
                    "prog_score": prog_score,
                    "llm_score": llm_score,
                    "reason": combined_reason,
                    "status": status,
                    "tool_calls": summary.get("tool_calls", 0),
                    "llm_calls": summary.get("llm_calls", 0),
                    "tools_used": summary.get("tools_used", []),
                }
            )

            print(f"    {status} | Score: {combined_score:.2f} (Prog: {prog_score:.2f}, LLM: {llm_score:.2f})")
            print(f"    Tools: {summary.get('tool_calls', 0)} | LLM Calls: {summary.get('llm_calls', 0)} | {prog_reason}")

        except Exception as e:  # noqa: BLE001
            print(f"    ERROR during evaluation: {type(e).__name__}: {e}")
            # Fall back to programmatic score only
            is_success = prog_score >= 0.8
            if is_success:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            if "AGENT ERROR" in test_case.actual_output:
                errors += 1
                status = "ERROR"

            results.append(
                {
                    "index": i + 1,
                    "input": GOLDENS[i].input,
                    "expected": GOLDENS[i].expected_output,
                    "actual": test_case.actual_output,
                    "score": prog_score,
                    "prog_score": prog_score,
                    "llm_score": 0.0,
                    "reason": f"Programmatic: {prog_reason}. LLM eval failed: {type(e).__name__}: {e!s}",
                    "status": status,
                    "tool_calls": summary.get("tool_calls", 0),
                    "llm_calls": summary.get("llm_calls", 0),
                    "tools_used": summary.get("tools_used", []),
                }
            )

    # Summary
    total = len(results)
    efficiency_rate = passed / total if total > 0 else 0.0

    # Calculate average tool calls and LLM calls
    avg_tool_calls = sum(r["tool_calls"] for r in results) / total if total > 0 else 0
    avg_llm_calls = sum(r["llm_calls"] for r in results) / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("STEP EFFICIENCY EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Cases: {total}")
    print(f"Passed:           {passed}")
    print(f"Failed:           {failed}")
    print(f"Errors:           {errors}")
    print(f"Efficiency Rate:  {efficiency_rate:.2%}")
    print(f"Avg Tool Calls:   {avg_tool_calls:.1f}")
    print(f"Avg LLM Calls:    {avg_llm_calls:.1f}")
    print("Threshold:        0.80")
    print(f"Overall:          {'PASS' if efficiency_rate >= 0.8 else 'FAIL'}")
    print("=" * 70)

    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 70)
    for r in results:
        print(f"\nTest {r['index']}: {r['status']} (score: {r['score']:.2f})")
        print(f"  Input:       {r['input'][:80]}...")
        print(f"  Tool Calls:  {r['tool_calls']} | LLM Calls: {r['llm_calls']} | Tools: {r['tools_used']}")
        print(f"  Prog Score:  {r['prog_score']:.2f} | LLM Score: {r['llm_score']:.2f}")
        print(f"  Reason:      {r['reason'][:150]}...")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "efficiency_rate": efficiency_rate,
        "avg_tool_calls": avg_tool_calls,
        "avg_llm_calls": avg_llm_calls,
        "results": results,
    }


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    run_step_efficiency_evaluation()