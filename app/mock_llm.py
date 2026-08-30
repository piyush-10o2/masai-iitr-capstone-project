"""
app/mock_llm.py — deterministic MOCK_LLM for CrewAI, via the brief's own
documented extension point: crewai.llms.base_llm.BaseLLM.

Two known pitfalls this file is deliberately designed to avoid:

1. CrewAI's built-in ReAct system-prompt template contains the literal
   example text "Observation: the result of the action". A parser that
   searches the FULL conversation (including the system message) for the
   substring "Observation:" will match this template text on the very first
   call, before any tool has run, and silently return placeholder text as a
   "final answer". This file only ever inspects the LAST non-system message
   to decide whether a real tool observation has come back — never the
   system prompt.

2. Dispatching a tool call by checking whether a keyword (e.g. "lookup") is a
   substring of the tool's NAME is fragile — a tool literally named
   `rag_lookup` would be silently misclassified. This file dispatches by
   inspecting each tool's own declared JSON-schema argument names instead.

The decision logic lives in MockLLMEngine, which has NO dependency on CrewAI
and is fully unit-testable offline (see the __main__ block below).
MockLLM is a thin adapter satisfying CrewAI's BaseLLM interface around it.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union


# ── Pure decision logic (no CrewAI dependency — fully unit-testable) ────────

TICKET_ID_PATTERN = re.compile(r"\bTCK-\d{4}\b")


class MockLLMEngine:
    """
    Deterministic decision logic for MOCK_LLM. Given the current message
    history and the list of available tool schemas, decides whether to:
      (a) call a tool (returns a ReAct-formatted Action/Action Input string), or
      (b) give a Final Answer (once a real tool Observation is present).
    """

    def decide(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        observation = self._extract_real_observation(messages)
        if observation is not None:
            return self._final_answer_from_observation(observation)

        user_query = self._latest_user_content(messages)
        tool_name, tool_args = self._select_tool_and_args(user_query, tools)
        return self._format_action(tool_name, tool_args)

    # -- Pitfall #1 fix: only ever look at the LAST non-system message -------
    def _extract_real_observation(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Returns the text of a genuine tool observation ONLY if the most
        recent message in the conversation actually is one. Never greps the
        whole conversation for "Observation:" — that would match the system
        prompt's own "Observation: the result of the action" template text
        on turn one, before any tool has run.
        """
        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system:
            return None
        last = non_system[-1]
        content = last.get("content", "")
        if last.get("role") in ("user", "tool", "function") and content.strip().startswith("Observation:"):
            return content.strip()[len("Observation:"):].strip()
        return None

    def _latest_user_content(self, messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    # -- Pitfall #2 fix: dispatch by declared argument schema, not name ------
    def _select_tool_and_args(self, user_query: str, tools: List[Dict[str, Any]]) -> Tuple[str, dict]:
        """
        Picks a tool by inspecting each tool's declared JSON-schema parameter
        names — NEVER by checking whether a keyword like "lookup" is a
        substring of the tool's NAME (a tool literally named `rag_lookup`
        would be silently misclassified by that approach).
        """
        ticket_match = TICKET_ID_PATTERN.search(user_query)

        record_id_tool = None
        query_tool = None
        for tool in tools:
            fn = tool.get("function", tool)  # tolerate wrapped or bare schema
            params_schema = fn.get("parameters", {})
            params = params_schema.get("properties", {})
            required = params_schema.get("required", list(params.keys()))
            if "record_id" in params or "record_id" in required:
                record_id_tool = fn["name"]
            if any(p in params for p in ("query", "question", "text")):
                query_tool = fn["name"]

        if ticket_match and record_id_tool:
            return record_id_tool, {"record_id": ticket_match.group(0)}
        if query_tool:
            return query_tool, {"query": user_query}

        raise ValueError(
            f"MockLLMEngine could not match query {user_query!r} to any tool's "
            f"declared argument schema. Available tools: "
            f"{[t.get('function', t).get('name') for t in tools]}"
        )

    def _format_action(self, tool_name: str, tool_args: dict) -> str:
        return (
            "Thought: I need to use a tool to answer this.\n"
            f"Action: {tool_name}\n"
            f"Action Input: {json.dumps(tool_args)}"
        )

    def _final_answer_from_observation(self, observation: str) -> str:
        return (
            "Thought: I now have the information I need.\n"
            f"Final Answer: {observation}"
        )


# ── Thin CrewAI adapter ──────────────────────────────────────────────────────

try:
    from crewai.llms.base_llm import BaseLLM
except ImportError:
    # Allows MockLLMEngine (and this adapter) to be unit-tested without
    # CrewAI installed. In your real environment, crewai IS installed and
    # this branch is never taken.
    class BaseLLM:  # type: ignore
        def __init__(self, model: str, temperature: Optional[float] = None,
                     stop: Optional[List[str]] = None):
            self.model = model
            self.temperature = temperature
            self.stop = stop or []


class MockLLM(BaseLLM):
    """
    Deterministic, keyless, network-free LLM for CrewAI's BaseLLM extension
    point. All decision logic lives in MockLLMEngine above; this class only
    adapts that logic to CrewAI's expected call() signature.
    """

    def __init__(self, model: str = "mock-llm", **kwargs):
        super().__init__(model=model, **kwargs)
        self.engine = MockLLMEngine()

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.engine.decide(messages, tools or [])

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192


if __name__ == "__main__":
    # Offline smoke test for BOTH documented pitfalls — no CrewAI needed.
    # Run: python -m app.mock_llm
    engine = MockLLMEngine()

    SYSTEM_PROMPT_WITH_TEMPLATE_TEXT = (
        "You are a helpful agent. Use the following format:\n"
        "Thought: your reasoning\n"
        "Action: the tool to use\n"
        "Action Input: the input to the tool\n"
        "Observation: the result of the action\n"   # <- the literal trap text
        "... (repeat as needed)\n"
        "Final Answer: the final answer"
    )

    TOOLS = [
        {"function": {"name": "rag_lookup",
                      "parameters": {"properties": {"query": {}}, "required": ["query"]}}},
        {"function": {"name": "check_support_ticket_status",
                      "parameters": {"properties": {"record_id": {}}, "required": ["record_id"]}}},
    ]

    print("=== Test 1: pitfall #1 — system-prompt template text ===")
    turn1_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_WITH_TEMPLATE_TEXT},
        {"role": "user", "content": "What is Ola's refund policy for cancelled rides?"},
    ]
    result1 = engine.decide(turn1_messages, TOOLS)
    print(result1)
    assert "Final Answer" not in result1, "BUG: matched the system prompt's own template text!"
    print("PASS: did not prematurely produce a Final Answer on turn 1.\n")

    print("=== Test 2: pitfall #2 — dispatch by schema, not name substring ===")
    turn1_ticket_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_WITH_TEMPLATE_TEXT},
        {"role": "user", "content": "What's the status of ticket TCK-0007?"},
    ]
    result2 = engine.decide(turn1_ticket_messages, TOOLS)
    print(result2)
    assert "check_support_ticket_status" in result2, "BUG: should dispatch to the ticket-lookup tool"
    assert "rag_lookup" not in result2
    print("PASS: correctly dispatched to check_support_ticket_status despite 'rag_lookup' "
          "also containing the substring 'lookup'.\n")

    print("=== Test 3: a real observation correctly produces a Final Answer ===")
    turn2_messages = turn1_ticket_messages + [
        {"role": "assistant", "content": result2},
        {"role": "user", "content": "Observation: {'status': 'Escalated', 'escalation_score': 0.84}"},
    ]
    result3 = engine.decide(turn2_messages, TOOLS)
    print(result3)
    assert "Final Answer" in result3
    assert "escalation_score" in result3
    print("PASS: correctly produced a grounded Final Answer from the real observation.")