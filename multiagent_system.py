# multiagent_system.py
from __future__ import annotations

from typing import Any, Dict

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Agents & state
from agents import (
    AgentState,
    OrchestratorAgent,
    GeneratorAgent,
    EvaluatorAgent,
)
# LLM + RAG context
from utils import get_llm
from retriever.get_context import get_context
from agents.evaluator import AFFINA_BASE_TEMPLATE


class MultiAgentSystem:
    def __init__(self):
        llm = get_llm(model="qwen3:1.7b", temperature=0.3, verbose=True)

        self.orchestrator = OrchestratorAgent(llm)
        self.generator = GeneratorAgent(llm)
        self.evaluator = EvaluatorAgent(llm, AFFINA_BASE_TEMPLATE)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
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

        # Nhánh điều kiện: tiếp tục hay kết thúc
        workflow.add_conditional_edges(
            "evaluator",
            self.should_continue_generation,
            {"continue": "generator", "finalize": "finalize"},
        )
        workflow.add_edge("finalize", END)

        # Checkpointer (in-memory)
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def initialize_node(self, state: AgentState) -> AgentState:
        print("🚀 Khởi tạo hệ thống Multi-Agent RAG")
        ctx = get_context(state["user_request"], top_k=5)

        return {
            **state,
            "context_segments": ctx,
            "iteration": 0,
            "max_iterations": state.get("max_iterations", 3),
            "pass_threshold": state.get("pass_threshold", 0.8),
            "best_result": {"score": 0.0, "content": ""},
            "thinking_log": [],
            "feedback": "",
            "should_continue": True,
        }

    def orchestrator_node(self, state: AgentState) -> AgentState:
        print("📋 Orchestrator Agent …")
        plan = self.orchestrator.plan(state["user_request"])
        print("── Kế hoạch:\n", plan["plan"][:300], "…")
        return {**state, "orchestrator_plan": plan}

    def generator_node(self, state: AgentState) -> AgentState:
        print(f"🔸 Generator vòng {state['iteration'] + 1}")
        gen_out = self.generator.generate(
            state["user_request"],
            state["orchestrator_plan"],
            state["context_segments"],
            state["feedback"],
        )
        return {**state, "generator_output": gen_out, "iteration": state["iteration"] + 1}

    def evaluator_node(self, state: AgentState) -> AgentState:
        print(f"⚖️  Evaluator vòng {state['iteration']}")
        gen_out = state["generator_output"]

        # Quick gate: nội dung quá ngắn
        if not gen_out["content"] or len(gen_out["content"].strip()) < 100:
            eval_out = {
                "score": 0.0,
                "feedback": "Template quá ngắn hoặc thiếu format. Hãy in đầy đủ đúng cấu trúc bảng.",
                "thinking": "Content quá ngắn",
            }
        else:
            eval_out = self.evaluator.evaluate(gen_out)

        # Log thinking
        thinking_entry = {
            "iteration": state["iteration"],
            "generator_thinking": gen_out.get("thinking", ""),
            "evaluator_thinking": eval_out.get("thinking", ""),
            "score": eval_out["score"],
            "feedback": eval_out["feedback"],
        }
        best = state["best_result"]
        if eval_out["score"] > best["score"]:
            best = {"score": eval_out["score"], "content": gen_out["content"]}

        print(f"📊 Score: {eval_out['score']:.2f}")

        return {
            **state,
            "evaluator_output": eval_out,
            "thinking_log": state["thinking_log"] + [thinking_entry],
            "best_result": best,
            "feedback": eval_out["feedback"],
        }

    @staticmethod
    def should_continue_generation(state: AgentState) -> str:
        if (
            state["evaluator_output"]["score"] >= state["pass_threshold"]
            and state["generator_output"]["content"]
            and len(state["generator_output"]["content"].strip()) > 100
        ):
            print("✅ Đạt chuẩn – dừng.")
            return "finalize"

        if state["iteration"] >= state["max_iterations"]:
            print(f"⚠️ Hết {state['max_iterations']} vòng lặp.")
            return "finalize"

        print("🔄 Tiếp tục sinh mới …")
        return "continue"

    def finalize_node(self, state: AgentState) -> AgentState:
        print("🏁 Hoàn tất.")
        return {**state, "final_result": state["best_result"]["content"], "should_continue": False}

    def run(
        self,
        user_request: str,
        max_iterations: int = 3,
        pass_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        init_state = AgentState(
            user_request=user_request,
            context_segments=[],
            orchestrator_plan={},
            generator_output={},
            evaluator_output={},
            feedback="",
            iteration=0,
            max_iterations=max_iterations,
            pass_threshold=pass_threshold,
            best_result={"score": 0.0, "content": ""},
            thinking_log=[],
            final_result=None,
            should_continue=True,
        )

        config = {"configurable": {"thread_id": "main_thread"}}
        result = self.graph.invoke(init_state, config=config)

        return {
            "content": result["final_result"],
            "score": result["best_result"]["score"],
            "orchestrator_plan": result["orchestrator_plan"],
            "thinking_log": result["thinking_log"],
            "iterations": result["iteration"],
        }


if __name__ == "__main__":
    system = MultiAgentSystem()
    out = system.run("Tạo chương trình khuyến mãi AFFINA cho tháng 9 về bảo hiểm sức khỏe")
    print("\n=== KẾT QUẢ ===")
    print("Score:", out["score"])
    print(out["content"])
