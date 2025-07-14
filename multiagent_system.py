from __future__ import annotations

import json
from config import CONFIG
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents import (
    AgentState,
    OrchestratorAgent,
    GeneratorAgent,
    EvaluatorAgent,
)

from utils import get_llm
from utils.save_to_word import save_to_word


class MultiAgentSystem:
    def __init__(self):
        self.llm = get_llm(model="qwen3:1.7b", temperature=0.7, verbose=True)
        self.tavily_api_key = CONFIG.get("TAVILY_API_KEY")
        
        if not self.tavily_api_key:
            self._log("WARNING: Tavily API key not found. Web search will be disabled.")

        self.orchestrator = OrchestratorAgent(
            llm=self.llm,
            tavily_api_key=self.tavily_api_key
        )
        self.generator = GeneratorAgent(llm=self.llm)
        self.evaluator = EvaluatorAgent(self.llm)
        self.graph = self._build_graph()

    def _log(self, message: str, level: str = "INFO"):
        """Centralized logging function"""
        print(f"[{level}] {message}")

    def _log_json(self, data: Dict[str, Any], title: str = "Data"):
        """Log JSON data in a formatted way"""
        print(f"\n=== {title} ===")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("=" * (len(title) + 8))

    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)

        workflow.add_node("initialize", self.initialize_node)
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("generator", self.generator_node)
        workflow.add_node("evaluator", self.evaluator_node)
        workflow.add_node("finalize", self.finalize_node)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "orchestrator")
        workflow.add_edge("orchestrator", "generator")
        workflow.add_edge("generator", "evaluator")

        workflow.add_conditional_edges(
            "evaluator",
            self.should_continue_generation,
            {
                "continue": "generator",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("finalize", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def initialize_node(self, state: AgentState) -> AgentState:
        """Initialize system state"""
        self._log("Initializing Multi-Agent System")

        if not state.get("custom_criteria"):
            state["custom_criteria"] = self.evaluator.get_default_criteria()

        initialized_state = {
            **state,
            "iteration": 0,
            "max_iterations": state.get("max_iterations", 3),
            "pass_threshold": state.get("pass_threshold", 0.75),
            "best_result": {"score": 0.0, "content": ""},
            "thinking_log": [],
            "feedback": "",
            "should_continue": True,
            "enable_search": state.get("enable_search", True),
            "search_results": None,
            "search_content": None
        }

        self._log_json({
            "user_request": state["user_request"],
            "language": state.get("language", "vietnamese"),
            "topic_type": state.get("topic_type", "food_nutrition"),
            "target_audience": state.get("target_audience"),
            "max_iterations": initialized_state["max_iterations"],
            "pass_threshold": initialized_state["pass_threshold"],
            "enable_search": initialized_state["enable_search"]
        }, "INITIALIZATION CONFIG")

        return initialized_state

    def orchestrator_node(self, state: AgentState) -> AgentState:
        """Process Orchestrator with Tavily Search"""
        self._log("Running Orchestrator Agent")

        try:
            plan_result = self.orchestrator.plan(
                user_request=state["user_request"],
                language=state.get("language", "vietnamese"),
                topic_type=state.get("topic_type", "food_nutrition"),
                target_audience=state.get("target_audience"),
                custom_hashtags=state.get("custom_hashtags"),
                enable_search=state.get("enable_search", True)
            )
            
            self._log("Orchestrator plan completed successfully")
            
            # Log orchestrator planning details
            self._log_json({
                "plan": plan_result.get("plan", ""),
                "thinking": plan_result.get("thinking", ""),
                "language": plan_result.get("language"),
                "topic_type": plan_result.get("topic_type"),
                "target_audience": plan_result.get("target_audience"),
                "custom_hashtags": plan_result.get("custom_hashtags")
            }, "ORCHESTRATOR PLANNING")
            
            # Log search results if available
            if plan_result.get("search_results"):
                search_results = plan_result["search_results"]
                self._log_json({
                    "total_results": search_results.get("total_results", 0),
                    "query": search_results.get("query", ""),
                    "answer": search_results.get("answer", "")[:500] + "..." if search_results.get("answer") else "",
                    "sources_count": len(search_results.get("results", [])),
                    "sources": [
                        {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "score": result.get("score", 0)
                        } for result in search_results.get("results", [])[:3]
                    ]
                }, "TAVILY SEARCH RESULTS")
            
            return {
                **state,
                "orchestrator_plan": plan_result,
                "search_results": plan_result.get("search_results")
            }
            
        except Exception as e:
            self._log(f"Orchestrator error: {str(e)}", "ERROR")
            fallback_plan = {
                "plan": f"Create content for request: {state['user_request']}",
                "thinking": f"Error in analysis: {str(e)}",
                "language": state.get("language", "vietnamese"),
                "topic_type": state.get("topic_type", "food_nutrition"),
                "target_audience": state.get("target_audience"),
                "custom_hashtags": state.get("custom_hashtags"),
                "search_results": None
            }
            return {
                **state,
                "orchestrator_plan": fallback_plan,
                "search_results": None
            }

    def generator_node(self, state: AgentState) -> AgentState:
        """Process Generator using search content from Orchestrator"""
        iteration = state['iteration'] + 1
        self._log(f"Running Generator Agent - Iteration {iteration}")

        try:
            orchestrator_plan = state["orchestrator_plan"]
            
            gen_result = self.generator.generate(
                user_request=state["user_request"],
                plan_data=orchestrator_plan,
                language=state.get("language", "vietnamese"),
                post_type=self._map_topic_to_post_type(state.get("topic_type", "food_nutrition")),
                target_audience=state.get("target_audience"),
                custom_hashtags=state.get("custom_hashtags"),
                feedback=state["feedback"]
            )
            
            self._log(f"Generator completed - Content length: {len(gen_result.get('content', ''))}")
            
            # Log generator details
            self._log_json({
                "thinking": gen_result.get("thinking", ""),
                "content_length": len(gen_result.get("content", "")),
                "content_preview": gen_result.get("content", "")[:200] + "..." if gen_result.get("content") else "",
                "language": gen_result.get("language"),
                "post_type": gen_result.get("post_type"),
                "target_audience": gen_result.get("target_audience"),
                "custom_hashtags": gen_result.get("custom_hashtags"),
                "search_content_used": bool(gen_result.get("search_content"))
            }, f"GENERATOR OUTPUT - ITERATION {iteration}")
            
            if gen_result.get("search_content"):
                self._log(f"Using search content from Orchestrator: {len(gen_result['search_content'])} characters")
            
            return {
                **state,
                "generator_output": gen_result,
                "search_content": gen_result.get("search_content"),
                "iteration": iteration
            }
            
        except Exception as e:
            self._log(f"Generator error: {str(e)}", "ERROR")
            fallback_gen = {
                "content": f"Content generated for request: {state['user_request']}",
                "thinking": f"Error in content generation: {str(e)}",
                "language": state.get("language", "vietnamese"),
                "post_type": self._map_topic_to_post_type(state.get("topic_type", "food_nutrition")),
                "target_audience": state.get("target_audience"),
                "custom_hashtags": state.get("custom_hashtags"),
                "search_content": None
            }
            return {
                **state,
                "generator_output": fallback_gen,
                "search_content": None,
                "iteration": iteration
            }

    def evaluator_node(self, state: AgentState) -> AgentState:
        """Process Evaluator with detailed feedback"""
        self._log(f"Running Evaluator Agent - Iteration {state['iteration']}")

        gen_output = state["generator_output"]

        if not gen_output.get("content") or len(gen_output["content"].strip()) < 50:
            eval_result = {
                "score": 0.0,
                "feedback": "Content is too short or lacks information. Please generate more complete content.",
                "thinking": "Content is too short or missing"
            }
            self._log("Content too short for evaluation", "WARNING")
        else:
            try:
                post_type = self._map_topic_to_post_type(state.get("topic_type", "food_nutrition"))
                custom_criteria = state.get("custom_criteria") or self.evaluator.get_default_criteria()
                
                eval_result = self.evaluator.evaluate(
                    candidate=gen_output,
                    post_type=post_type,
                    language=state.get("language", "vietnamese"),
                    target_audience=state.get("target_audience"),
                    custom_criteria=custom_criteria,
                    evaluation_focus=state.get("evaluation_focus")
                )
                
            except Exception as e:
                self._log(f"Evaluator error: {str(e)}", "ERROR")
                eval_result = {
                    "score": 0.3,
                    "feedback": f"Error in evaluation: {str(e)}. Content needs improvement.",
                    "thinking": f"Evaluation error: {str(e)}"
                }

        # Log evaluation details
        self._log_json({
            "iteration": state["iteration"],
            "score": eval_result["score"],
            "feedback": eval_result["feedback"],
            "thinking": eval_result.get("thinking", ""),
            "criteria_used": state.get("custom_criteria", {}),
            "content_length": len(gen_output.get("content", "")),
            "pass_threshold": state["pass_threshold"]
        }, f"EVALUATOR FEEDBACK - ITERATION {state['iteration']}")

        thinking_entry = {
            "iteration": state["iteration"],
            "generator_thinking": gen_output.get("thinking", ""),
            "evaluator_thinking": eval_result.get("thinking", ""),
            "score": eval_result["score"],
            "feedback": eval_result["feedback"]
        }

        new_thinking_log = state["thinking_log"] + [thinking_entry]

        best_result = state["best_result"]
        if eval_result["score"] > best_result["score"]:
            best_result = {
                "score": eval_result["score"],
                "content": gen_output["content"],
                "iteration": state["iteration"]
            }
            self._log(f"NEW BEST RESULT! Score: {eval_result['score']:.3f}")

        self._log(f"Current Score: {eval_result['score']:.3f}")
        if eval_result["feedback"]:
            self._log(f"Feedback: {eval_result['feedback']}")

        return {
            **state,
            "evaluator_output": eval_result,
            "thinking_log": new_thinking_log,
            "best_result": best_result,
            "feedback": eval_result["feedback"]
        }

    def should_continue_generation(self, state: AgentState) -> str:
        """Decide whether to continue generation"""
        eval_output = state["evaluator_output"]

        if (eval_output["score"] >= state["pass_threshold"] and
            state["generator_output"].get("content") and
            len(state["generator_output"]["content"].strip()) > 50):
            self._log(f"Threshold reached: {eval_output['score']:.3f} >= {state['pass_threshold']}")
            return "finalize"

        if state["iteration"] >= state["max_iterations"]:
            self._log(f"Maximum iterations reached: {state['max_iterations']}")
            return "finalize"

        self._log(f"Continuing - Score: {eval_output['score']:.3f}")
        return "continue"

    def finalize_node(self, state: AgentState) -> AgentState:
        """Finalize results"""
        self._log("Finalizing results")

        final_result = state["best_result"]["content"]
        final_score = state["best_result"]["score"]

        docx_path = None
        try:
            docx_path = save_to_word(final_result, final_score)
            self._log(f"Saved to Word document: {docx_path}")
        except Exception as exc:
            self._log(f"Could not save Word document: {exc}", "WARNING")

        # Log final summary
        self._log_json({
            "final_score": final_score,
            "total_iterations": state["iteration"],
            "best_iteration": state["best_result"].get("iteration", "N/A"),
            "tavily_search_enabled": state.get("enable_search", False),
            "threshold_met": final_score >= state["pass_threshold"],
            "pass_threshold": state["pass_threshold"],
            "content_length": len(final_result),
            "docx_saved": docx_path is not None
        }, "FINAL SUMMARY")

        return {
            **state,
            "final_result": final_result,
            "docx_path": str(docx_path) if docx_path else None,
            "should_continue": False,
        }

    def _map_topic_to_post_type(self, topic_type: str) -> str:
        """Map topic_type to post_type"""
        topic_mapping = {
            "food_nutrition": "health_nutrition",
            "disease_warning": "disease_warning",
            "travel_adventure": "travel_adventure",
            "business_enterprise": "business_enterprise",
            "lifestyle_office": "lifestyle_office",
            "holiday_event": "holiday_event",
        }
        return topic_mapping.get(topic_type, "health_nutrition")

    def get_available_topics(self) -> List[str]:
        """Get available topic types"""
        return self.orchestrator.get_available_topics()

    def get_available_post_types(self) -> List[str]:
        """Get available post types"""
        return self.generator.get_available_post_types()

    def get_available_criteria(self) -> List[str]:
        """Get available criteria"""
        return self.evaluator.get_available_criteria()

    def get_default_criteria(self) -> Dict[str, float]:
        """Get default criteria"""
        return self.evaluator.get_default_criteria()

    def run(self, 
            user_request: str,
            language: str = "vietnamese",
            topic_type: str = "food_nutrition",
            target_audience: Optional[str] = None,
            custom_hashtags: Optional[List[str]] = None,
            custom_criteria: Optional[Dict[str, float]] = None,
            evaluation_focus: Optional[str] = None,
            max_iterations: int = 3,
            pass_threshold: float = 0.75,
            enable_search: bool = True,
            verbose: bool = True) -> Dict[str, Any]:
        
        # Validate inputs
        if language not in ["vietnamese", "english"]:
            raise ValueError("Language must be 'vietnamese' or 'english'")
        
        if topic_type not in self.get_available_topics():
            raise ValueError(f"Topic type must be one of: {self.get_available_topics()}")
        
        post_type = self._map_topic_to_post_type(topic_type)
        
        if custom_criteria is None:
            custom_criteria = self.get_default_criteria()

        from agents.state import create_initial_agent_state
        
        initial_state = create_initial_agent_state(
            user_request=user_request,
            language=language,
            topic_type=topic_type,
            target_audience=target_audience,
            custom_hashtags=custom_hashtags,
            custom_criteria=custom_criteria,
            evaluation_focus=evaluation_focus,
            max_iterations=max_iterations,
            pass_threshold=pass_threshold,
            enable_search=enable_search
        )

        if verbose:
            self._log_json({
                "user_request": user_request,
                "language": language,
                "topic_type": topic_type,
                "post_type": post_type,
                "target_audience": target_audience,
                "custom_hashtags": custom_hashtags,
                "pass_threshold": pass_threshold,
                "max_iterations": max_iterations,
                "tavily_search": enable_search
            }, "SYSTEM CONFIGURATION")

        config = {"configurable": {"thread_id": "main_thread"}}
        
        try:
            result = self.graph.invoke(initial_state, config=config)
            
            return {
                "content": result["final_result"],
                "score": result["best_result"]["score"],
                "orchestrator_plan": result["orchestrator_plan"],
                "search_results": result.get("search_results"),
                "search_content": result.get("search_content"),
                "thinking_log": result["thinking_log"],
                "iterations": result["iteration"],
                "docx_path": result.get("docx_path"),
                "best_iteration": result["best_result"].get("iteration"),
                "language": language,
                "topic_type": topic_type,
                "post_type": post_type,
                "target_audience": target_audience,
                "custom_hashtags": custom_hashtags,
                "custom_criteria": custom_criteria,
                "enable_search": enable_search,
                "success": result["best_result"]["score"] >= pass_threshold
            }
            
        except Exception as e:
            self._log(f"System error: {str(e)}", "ERROR")
            return {
                "content": f"Error: {str(e)}",
                "score": 0.0,
                "orchestrator_plan": {},
                "search_results": None,
                "search_content": None,
                "thinking_log": [],
                "iterations": 0,
                "docx_path": None,
                "error": str(e),
                "enable_search": enable_search,
                "success": False
            }


def main():
    """Main function to test the system"""
    system = MultiAgentSystem()
    
    test_cases = [
        {
            "user_request": "5 foods rich in magnesium that help improve sleep quality",
            "language": "english",
            "topic_type": "food_nutrition",
            "target_audience": "Everyone",
            "custom_hashtags": ["#nutrition", "#health", "#sleep"],
            "enable_search": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}")
        print(f"{'='*60}")
        
        result = system.run(**test_case)
        
        print(f"\nTEST CASE {i} RESULTS:")
        print(f"Success: {result['success']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Tavily search: {result['enable_search']}")
        print(f"Word file: {result.get('docx_path', 'None')}")
        
        if result.get('search_results'):
            search_stats = result['search_results']
            print(f"Search sources: {search_stats.get('total_results', 0)}")
        
        if result.get('search_content'):
            print(f"Search content: {len(result['search_content'])} characters")
        
        print(f"\nGenerated Content:\n{result['content']}")


if __name__ == "__main__":
    main()