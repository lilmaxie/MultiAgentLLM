from __future__ import annotations
from typing import List, Dict, Any, Optional, TypedDict


class AgentState(TypedDict):
    # Core request information
    user_request: str
    
    # Template configuration parameters
    language: str  # "vietnamese" or "english"
    topic_type: str  # for orchestrator (food_nutrition, disease_warning, etc.)
    post_type: str  # for generator/evaluator (health_nutrition, disease_warning, etc.)
    target_audience: Optional[str]  # Target audience description
    custom_hashtags: Optional[List[str]]  # Custom hashtags list
    custom_criteria: Optional[Dict[str, float]]  # Custom evaluation criteria with weights
    evaluation_focus: Optional[str]  # Specific focus for evaluation
    
    # Agent outputs
    orchestrator_plan: Dict[str, Any]  # Output from orchestrator agent
    generator_output: Dict[str, Any]  # Output from generator agent
    evaluator_output: Dict[str, Any]  # Output from evaluator agent
    
    # Feedback and iteration control
    feedback: str  # Feedback for next iteration
    iteration: int  # Current iteration number
    max_iterations: int  # Maximum allowed iterations
    pass_threshold: float  # Score threshold to pass
    
    # Results tracking
    best_result: Dict[str, Any]  # Best result so far (score, content, iteration)
    thinking_log: List[Dict[str, Any]]  # Log of all thinking processes
    final_result: Optional[str]  # Final generated content
    docx_path: Optional[str]  # Path to saved Word document
    
    # Control flow
    should_continue: bool  # Whether to continue the workflow


# Optional: Helper class for better type hints in specific contexts
class OrchestratorOutput(TypedDict):
    thinking: str
    plan: str
    full_response: str
    template_variables: Dict[str, Any]
    language: str
    topic_type: str
    target_audience: Optional[str]
    custom_hashtags: Optional[List[str]]


class GeneratorOutput(TypedDict):
    thinking: str
    content: str
    full_response: str
    template_variables: Dict[str, Any]
    language: str
    post_type: str
    target_audience: Optional[str]
    custom_hashtags: Optional[List[str]]


class EvaluatorOutput(TypedDict):
    thinking: str
    score: float
    feedback: str
    full_response: str
    template_variables: Dict[str, Any]
    language: str
    post_type: str
    target_audience: Optional[str]
    custom_criteria: Optional[Dict[str, float]]


class BestResult(TypedDict):
    score: float
    content: str
    iteration: Optional[int]


class ThinkingLogEntry(TypedDict):
    iteration: int
    generator_thinking: str
    evaluator_thinking: str
    score: float
    feedback: str


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

# Default values for optional parameters
DEFAULT_VALUES = {
    "language": "vietnamese",
    "topic_type": "food_nutrition",
    "post_type": "health_nutrition",
    "target_audience": None,
    "custom_hashtags": [],
    "custom_criteria": None,
    "evaluation_focus": None,
    "max_iterations": 3,
    "pass_threshold": 0.75,
    "feedback": "",
    "iteration": 0,
    "should_continue": True
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
    pass_threshold: float = 0.75
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
        "total_thinking_entries": len(state["thinking_log"])
    }