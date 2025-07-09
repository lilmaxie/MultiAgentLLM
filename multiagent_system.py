# multiagent_system.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

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
# LLM + utilities
from utils import get_llm
from utils.save_to_word import save_to_word


class MultiAgentSystem:
    def __init__(self):
        llm = get_llm(model="qwen3:1.7b", temperature=0.8, verbose=True)

        self.orchestrator = OrchestratorAgent(llm)
        self.generator = GeneratorAgent(llm)
        self.evaluator = EvaluatorAgent(llm)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Xây dựng LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Thêm các nodes
        workflow.add_node("initialize", self.initialize_node)
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("generator", self.generator_node)
        workflow.add_node("evaluator", self.evaluator_node)
        workflow.add_node("finalize", self.finalize_node)

        # Định nghĩa edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "orchestrator")
        workflow.add_edge("orchestrator", "generator")
        workflow.add_edge("generator", "evaluator")

        # Conditional edge: tiếp tục hoặc kết thúc
        workflow.add_conditional_edges(
            "evaluator",
            self.should_continue_generation,
            {
                "continue": "generator",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("finalize", END)

        # Sử dụng checkpointer với cấu hình thread_id
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def initialize_node(self, state: AgentState) -> AgentState:
        """Khởi tạo state ban đầu"""
        print("🚀 Khởi tạo hệ thống Multi-Agent với Jinja2 Templates")

        # Ensure default criteria is set if not provided
        if not state.get("custom_criteria"):
            state["custom_criteria"] = self.evaluator.get_default_criteria()

        return {
            **state,
            "iteration": 0,
            "max_iterations": state.get("max_iterations", 3),
            "pass_threshold": state.get("pass_threshold", 0.75),
            "best_result": {"score": 0.0, "content": ""},
            "thinking_log": [],
            "feedback": "",
            "should_continue": True
        }

    def orchestrator_node(self, state: AgentState) -> AgentState:
        """Node xử lý Orchestrator với Jinja2 template"""
        print("📋 Chạy Orchestrator Agent...")

        try:
            # Lấy các parameters từ state
            plan_result = self.orchestrator.plan(
                user_request=state["user_request"],
                language=state.get("language", "vietnamese"),
                topic_type=state.get("topic_type", "food_nutrition"),
                target_audience=state.get("target_audience"),
                custom_hashtags=state.get("custom_hashtags")
            )
            
            print(f"✅ Orchestrator plan completed")
            if plan_result.get("thinking"):
                print(f"💭 Thinking: {plan_result['thinking']}")
            
            return {
                **state,
                "orchestrator_plan": plan_result
            }
            
        except Exception as e:
            print(f"❌ Orchestrator error: {str(e)}")
            # Fallback plan
            fallback_plan = {
                "plan": f"Tạo nội dung theo yêu cầu: {state['user_request']}",
                "thinking": f"Lỗi trong quá trình phân tích: {str(e)}",
                "language": state.get("language", "vietnamese"),
                "topic_type": state.get("topic_type", "food_nutrition"),
                "target_audience": state.get("target_audience"),
                "custom_hashtags": state.get("custom_hashtags")
            }
            return {
                **state,
                "orchestrator_plan": fallback_plan
            }

    def generator_node(self, state: AgentState) -> AgentState:
        """Node xử lý Generator với Jinja2 template"""
        print(f"🔸 Generator - Lần lặp {state['iteration'] + 1}")

        try:
            # Lấy thông tin từ orchestrator plan
            orchestrator_plan = state["orchestrator_plan"]
            
            gen_result = self.generator.generate(
                user_request=state["user_request"],
                plan_data=orchestrator_plan,
                language=state.get("language", "vietnamese"),
                post_type=state.get("post_type", "health_nutrition"),  # Map topic_type to post_type
                target_audience=state.get("target_audience"),
                custom_hashtags=state.get("custom_hashtags"),
                feedback=state["feedback"]
            )
            
            print(f"✅ Generator completed - Content length: {len(gen_result.get('content', ''))}")
            
            return {
                **state,
                "generator_output": gen_result,
                "iteration": state["iteration"] + 1
            }
            
        except Exception as e:
            print(f"❌ Generator error: {str(e)}")
            # Fallback generation
            fallback_gen = {
                "content": f"Nội dung được tạo cho yêu cầu: {state['user_request']}",
                "thinking": f"Lỗi trong quá trình tạo nội dung: {str(e)}",
                "language": state.get("language", "vietnamese"),
                "post_type": state.get("post_type", "health_nutrition"),
                "target_audience": state.get("target_audience"),
                "custom_hashtags": state.get("custom_hashtags")
            }
            return {
                **state,
                "generator_output": fallback_gen,
                "iteration": state["iteration"] + 1
            }

    def evaluator_node(self, state: AgentState) -> AgentState:
        """Node xử lý Evaluator với Jinja2 template"""
        print(f"⚖️ Evaluator - Đánh giá lần {state['iteration']}")

        gen_output = state["generator_output"]

        # Kiểm tra nếu content quá ngắn
        if not gen_output.get("content") or len(gen_output["content"].strip()) < 50:
            eval_result = {
                "score": 0.0,
                "feedback": "Nội dung quá ngắn hoặc thiếu thông tin. Hãy tạo nội dung đầy đủ hơn.",
                "thinking": "Content quá ngắn hoặc không có"
            }
        else:
            try:
                # Map post_type if needed
                post_type = self._map_topic_to_post_type(state.get("topic_type", "food_nutrition"))
                
                # Ensure custom_criteria is provided
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
                print(f"❌ Evaluator error: {str(e)}")
                eval_result = {
                    "score": 0.3,
                    "feedback": f"Lỗi trong quá trình đánh giá: {str(e)}. Nội dung cần cải thiện.",
                    "thinking": f"Evaluation error: {str(e)}"
                }

        # Cập nhật thinking log
        thinking_entry = {
            "iteration": state["iteration"],
            "generator_thinking": gen_output.get("thinking", ""),
            "evaluator_thinking": eval_result.get("thinking", ""),
            "score": eval_result["score"],
            "feedback": eval_result["feedback"]
        }

        new_thinking_log = state["thinking_log"] + [thinking_entry]

        # Cập nhật best result nếu tốt hơn
        best_result = state["best_result"]
        if eval_result["score"] > best_result["score"]:
            best_result = {
                "score": eval_result["score"],
                "content": gen_output["content"],
                "iteration": state["iteration"]
            }

        print(f"📊 Score: {eval_result['score']:.2f}")
        if eval_result["feedback"]:
            print(f"💬 Feedback: {eval_result['feedback']}")

        return {
            **state,
            "evaluator_output": eval_result,
            "thinking_log": new_thinking_log,
            "best_result": best_result,
            "feedback": eval_result["feedback"]
        }

    def should_continue_generation(self, state: AgentState) -> str:
        """Quyết định có tiếp tục generate hay không"""
        eval_output = state["evaluator_output"]

        # Kiểm tra điều kiện dừng
        if (eval_output["score"] >= state["pass_threshold"] and
            state["generator_output"].get("content") and
            len(state["generator_output"]["content"].strip()) > 50):
            print(f"✅ Đạt chuẩn: {eval_output['score']:.2f} ≥ {state['pass_threshold']}")
            return "finalize"

        # Kiểm tra số lần lặp tối đa
        if state["iteration"] >= state["max_iterations"]:
            print(f"⚠️ Hết {state['max_iterations']} vòng lặp")
            return "finalize"

        print(f"🔄 Tiếp tục - Score: {eval_output['score']:.2f}")
        return "continue"

    def finalize_node(self, state: AgentState) -> AgentState:
        """Node hoàn thiện kết quả cuối cùng"""
        print("🏁 Hoàn thiện kết quả…")

        final_result = state["best_result"]["content"]
        final_score = state["best_result"]["score"]

        # Lưu file Word
        docx_path = None
        try:
            docx_path = save_to_word(final_result, final_score)
            print(f"📄 Đã lưu bài viết ra: {docx_path}")
        except Exception as exc:
            print(f"⚠️ Không thể ghi file Word: {exc}")

        print(f"📊 Điểm số cuối cùng: {final_score:.2f}")
        print(f"🔄 Số lần lặp: {state['iteration']}")
        print(f"🎯 Lần lặp tốt nhất: {state['best_result'].get('iteration', 'N/A')}")
        print(
            f"✅ Trạng thái: {'ĐẠT CHUẨN' if final_score >= state['pass_threshold'] else 'CHƯA ĐẠT CHUẨN'}"
        )

        return {
            **state,
            "final_result": final_result,
            "docx_path": str(docx_path) if docx_path else None,
            "should_continue": False,
        }

    def _map_topic_to_post_type(self, topic_type: str) -> str:
        """Map topic_type từ orchestrator sang post_type cho generator/evaluator"""
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
        """Lấy danh sách các topic types có sẵn"""
        return self.orchestrator.get_available_topics()

    def get_available_post_types(self) -> List[str]:
        """Lấy danh sách các post types có sẵn"""
        return self.generator.get_available_post_types()

    def get_available_criteria(self) -> List[str]:
        """Lấy danh sách các criteria có sẵn"""
        return self.evaluator.get_available_criteria()

    def get_default_criteria(self) -> Dict[str, float]:
        """Lấy criteria mặc định"""
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
            pass_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Chạy hệ thống multi-agent với input đầy đủ
        
        Args:
            user_request: Yêu cầu của người dùng
            language: Ngôn ngữ ("vietnamese" hoặc "english")
            topic_type: Loại chủ đề (food_nutrition, disease_warning, etc.)
            target_audience: Đối tượng mục tiêu
            custom_hashtags: Danh sách hashtags tùy chỉnh
            custom_criteria: Tiêu chí đánh giá tùy chỉnh
            evaluation_focus: Trọng tâm đánh giá
            max_iterations: Số lần lặp tối đa
            pass_threshold: Ngưỡng điểm để pass
        """
        
        # Validate inputs
        if language not in ["vietnamese", "english"]:
            raise ValueError("Language must be 'vietnamese' or 'english'")
        
        if topic_type not in self.get_available_topics():
            raise ValueError(f"Topic type must be one of: {self.get_available_topics()}")

        # Map topic_type to post_type
        post_type = self._map_topic_to_post_type(topic_type)

        # Use default criteria if none provided
        if custom_criteria is None:
            custom_criteria = self.get_default_criteria()

        # Import helper function
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
            pass_threshold=pass_threshold
        )

        print(f"🚀 Bắt đầu xử lý với cấu hình:")
        print(f"  📝 Yêu cầu: {user_request}")
        print(f"  🌐 Ngôn ngữ: {language}")
        print(f"  📂 Chủ đề: {topic_type} -> {post_type}")
        print(f"  👥 Đối tượng: {target_audience or 'Không xác định'}")
        print(f"  🏷️ Hashtags: {custom_hashtags or 'Mặc định'}")
        print(f"  📊 Criteria: {list(custom_criteria.keys()) if custom_criteria else 'Mặc định'}")
        print(f"  🎯 Ngưỡng pass: {pass_threshold}")
        print(f"  🔄 Tối đa {max_iterations} lần lặp")
        print("-" * 50)

        # Chạy graph với thread_id để sử dụng checkpointer
        config = {"configurable": {"thread_id": "main_thread"}}
        
        try:
            result = self.graph.invoke(initial_state, config=config)
            
            return {
                "content": result["final_result"],
                "score": result["best_result"]["score"],
                "orchestrator_plan": result["orchestrator_plan"],
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
                "success": result["best_result"]["score"] >= pass_threshold
            }
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình chạy hệ thống: {str(e)}")
            return {
                "content": f"Lỗi: {str(e)}",
                "score": 0.0,
                "orchestrator_plan": {},
                "thinking_log": [],
                "iterations": 0,
                "docx_path": None,
                "error": str(e),
                "success": False
            }


def main():
    """Hàm main để test hệ thống"""
    system = MultiAgentSystem()
    
    # Test với các tham số khác nhau
    test_cases = [
        # {
        #     "user_request": "5 lợi ích của việc ăn trái cây hàng ngày",
        #     "language": "vietnamese",
        #     "topic_type": "food_nutrition",
        #     "target_audience": "Người trung niên quan tâm đến sức khỏe",
        #     "custom_hashtags": ["#suckhoe", "#traicay", "#dinh_duong"]
        # },
        {
            "user_request": "Create a warning post about seasonal flu",
            "language": "english",
            "topic_type": "disease_warning",
            "target_audience": "General public",
            "custom_hashtags": ["#flu", "#health", "#prevention"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}")
        print(f"{'='*60}")
        
        result = system.run(**test_case)
        
        print(f"\n📊 KẾT QUẢ TEST CASE {i}:")
        print(f"✅ Thành công: {result['success']}")
        print(f"📊 Điểm số: {result['score']:.2f}")
        print(f"🔄 Số lần lặp: {result['iterations']}")
        print(f"📄 File Word: {result.get('docx_path', 'Không có')}")
        print(f"\n📝 Nội dung:\n{result['content']}")


if __name__ == "__main__":
    main()