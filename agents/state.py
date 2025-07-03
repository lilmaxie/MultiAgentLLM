from __future__ import annotations
from typing import List, Dict, Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- static ---
    max_iterations: int
    pass_threshold: float

    # --- dynamic ---
    user_request: str
    context_segments: List[str]
    orchestrator_plan: Dict[str, Any]
    generator_output: Dict[str, Any]
    evaluator_output: Dict[str, Any]
    feedback: str
    iteration: int
    best_result: Dict[str, Any]
    thinking_log: List[Dict[str, Any]]
    final_result: Optional[str]
    should_continue: bool
