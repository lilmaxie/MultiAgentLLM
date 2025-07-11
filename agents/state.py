from __future__ import annotations
from typing import List, Dict, Any, Optional, TypedDict


class AgentState(TypedDict):
    user_request: str
    
    language: str
    topic_type: str
    post_type: str
    target_audience: Optional[str]
    custom_hashtags: Optional[List[str]]
    custom_criteria: Optional[Dict[str, float]]
    evaluation_focus: Optional[str]
    
    # Web search configuration and results
    enable_search: bool
    search_results: Optional[Dict[str, Any]]
    
    # Agent outputs
    orchestrator_plan: Dict[str, Any]
    generator_output: Dict[str, Any]
    evaluator_output: Dict[str, Any]
    
    # Feedback and iteration control
    feedback: str
    iteration: int
    max_iterations: int
    pass_threshold: float
    
    # Results tracking
    best_result: Dict[str, Any]
    thinking_log: List[Dict[str, Any]]
    final_result: Optional[str]
    docx_path: Optional[str]
    
    # Control flow
    should_continue: bool


# Configuration validation helpers
VALID_LANGUAGES = ["vietnamese", "english"]

VALID_TOPIC_TYPES = [
    "food_nutrition",
    "disease_warning", 
    "travel_adventure",
    "business_enterprise",
    "lifestyle_office",
    "holiday_event"
]

VALID_POST_TYPES = [
    "health_nutrition",
    "disease_warning",
    "travel_adventure", 
    "business_enterprise",
    "lifestyle_office",
    "holiday_event"
]

# Topic type to post type mapping
TOPIC_TO_POST_TYPE_MAPPING = {
    "food_nutrition": "health_nutrition",
    "disease_warning": "disease_warning",
    "travel_adventure": "travel_adventure",
    "business_enterprise": "business_enterprise",
    "lifestyle_office": "lifestyle_office",
    "holiday_event": "holiday_event"
}


def validate_agent_state_params(
    language: str,
    topic_type: str,
    post_type: str
) -> None:
    """Validate AgentState parameters"""
    if language not in VALID_LANGUAGES:
        raise ValueError(f"Language must be one of: {VALID_LANGUAGES}")
    
    if topic_type not in VALID_TOPIC_TYPES:
        raise ValueError(f"Topic type must be one of: {VALID_TOPIC_TYPES}")
    
    if post_type not in VALID_POST_TYPES:
        raise ValueError(f"Post type must be one of: {VALID_POST_TYPES}")


def create_initial_agent_state(
    user_request: str,
    language: str = "vietnamese",
    topic_type: str = "food_nutrition",
    target_audience: Optional[str] = None,
    custom_hashtags: Optional[List[str]] = None,
    custom_criteria: Optional[Dict[str, float]] = None,
    evaluation_focus: Optional[str] = None,
    max_iterations: int = 3,
    pass_threshold: float = 0.75,
    enable_search: bool = True
) -> AgentState:
    """Create initial AgentState with validation"""
    
    # Map topic_type to post_type
    post_type = TOPIC_TO_POST_TYPE_MAPPING.get(topic_type, "health_nutrition")
    
    # Validate parameters
    validate_agent_state_params(language, topic_type, post_type)
    
    return AgentState(
        user_request=user_request,
        language=language,
        topic_type=topic_type,
        post_type=post_type,
        target_audience=target_audience,
        custom_hashtags=custom_hashtags or [],
        custom_criteria=custom_criteria,
        evaluation_focus=evaluation_focus,
        enable_search=enable_search,
        search_results=None,
        orchestrator_plan={},
        generator_output={},
        evaluator_output={},
        feedback="",
        iteration=0,
        max_iterations=max_iterations,
        pass_threshold=pass_threshold,
        best_result={"score": 0.0, "content": "", "iteration": None},
        thinking_log=[],
        final_result=None,
        docx_path=None,
        should_continue=True
    )


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """Get a summary of the current state for debugging"""
    return {
        "user_request": state["user_request"][:100] + "..." if len(state["user_request"]) > 100 else state["user_request"],
        "language": state["language"],
        "topic_type": state["topic_type"],
        "post_type": state["post_type"],
        "target_audience": state["target_audience"],
        "custom_hashtags": state["custom_hashtags"],
        "iteration": state["iteration"],
        "max_iterations": state["max_iterations"],
        "pass_threshold": state["pass_threshold"],
        "best_score": state["best_result"]["score"],
        "has_orchestrator_plan": bool(state["orchestrator_plan"]),
        "has_generator_output": bool(state["generator_output"]),
        "has_evaluator_output": bool(state["evaluator_output"]),
        "should_continue": state["should_continue"],
        "total_thinking_entries": len(state["thinking_log"]),
        "enable_search": state["enable_search"],
        "has_search_results": bool(state["search_results"])
    }